# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
Demo #13

The thirteenth script in our tutorial about using Galsim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script introduces non-idealities arising from NIR detectors, in particular those that will be
observed and accounted for in the Roman Space Telescope. Three such non-ideal effects are
demonstrated, in the order in which they are introduced in the detectors:

1) Reciprocity failure: Flux-dependent sensitivity of the detector.
2) Non-linearity: Charge-dependent gain in converting from units of electrons to ADU.  Non-linearity
   in some form is also relevant for CCDs in addition to NIR detectors.
3) Interpixel capacitance: Influence of charge in a pixel on the voltage reading of neighboring
   ones.

It also uses chromatic photon shooting, which is generally a more efficient way to simulate
scenes with many faint galaxies.  The default FFT method for drawing chromatically is fairly
slow, since it needs to integrate the image over the bandpass.  With photon shooting, the
photons are assigned wavelengths according to the SED of the galaxy, and then each photon has
the appropriate application of the chromatic PSF according to the wavelength.

This demo will by default produce 3 output images, one for each of the filters:
    Y106, J129, H158
To select a different set of Roman filters, you may use the `filters` option on the command line.
E.g. `python demo13.py -filters=ZJHF` will render Z087, J129, H158, and F184 images.

New features introduced in this demo:
- image.quantize()
- obj = galsim.DeltaFunction(flux)
- galsim.roman.addReciprocityFailure(image)
- galsim.roman.applyNonlinearity(image)
- galsim.roman.applyIPC(image)
- galsim.roman.getBandpasses()
- galsim.roman.getPSF()
- galsim.roman.getWCS()
- galsim.roman.allowedPos()
- galsim.roman.getSkyLevel()
"""

import argparse
import numpy as np
import sys, os
import math
import logging
import time
import galsim as galsim
import galsim.roman as roman
import datetime

def parse_args(argv):
    parser = argparse.ArgumentParser(prog='demo13', add_help=True)
    parser.add_argument('-f', '--filters', type=str, default='YJH', action='store',
                        help='Which filters to simulate (default = "YJH")')
    parser.add_argument('-o', '--outpath', type=str, default='output',
                        help='Which directory to put the output files')
    parser.add_argument('-n', '--ngal', type=int, default=400,
                        help='How many galaxies to draw')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Initial seed for random numbers')
    parser.add_argument('-s', '--sca', type=int, default=7, choices=range(1,19),
                        help='Which SCA to simulate (default is arbitrarily SCA 7)')
    parser.add_argument('--sample', type=str, default='23.5', choices=('23.5', '25.2', 'test'),
                        help='Which COSMOS sample to use')
    parser.add_argument('-v', '--verbosity', type=int, default=2, choices=range(0,4),
                        help='Verbosity level')

    args = parser.parse_args(argv)
    return args

def main(argv):

    args = parse_args(argv)
    use_filters = args.filters
    outpath = args.outpath
    ngal = args.ngal
    seed = args.seed
    use_SCA = args.sca
    sample = args.sample

    # Make output directory if not already present.
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    # Use a logger to output some information about the run.
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logger = logging.getLogger("demo13")
    logging_levels = { 0: logging.CRITICAL,
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    level = logging_levels[args.verbosity]
    logger.setLevel(level)

    # Read in the Roman filters, setting an AB zeropoint appropriate for this telescope given its
    # diameter and (since we didn't use any keyword arguments to modify this) using the typical
    # exposure time for Roman images.  By default, this routine truncates the parts of the
    # bandpasses that are near 0 at the edges, and thins them by the default amount.
    roman_filters = roman.getBandpasses(AB_zeropoint=True)
    logger.debug('Read in Roman imaging filters.')

    # Get the names of the ones we will use here.
    filters = [filter_name for filter_name in roman_filters if filter_name[0] in use_filters]
    logger.debug('Using filters: %s',filters)

    # Note: This example by default uses the full m<23.5 COSMOS sample, rather than the smaller
    #       sample that demo 11 used.
    #       You can use the galsim_download_cosmos script to download it.
    #       We recommend specifying the directory for the download, rather than let it use the
    #       default directory, since that will be in the GalSim python share directory, which will
    #       be overwritten whenever you reinstall GalSim.  This command sets up a symlink from that
    #       location to a directory in your home directory.  (Feel free to use any other convenient
    #       directory of course, depending on your situation.)
    #
    #           galsim_download_cosmos -s 23.5 -d ~/cosmos
    #
    # The area and exposure time here rescale the fluxes to be appropriate for the Roman collecting
    # area and exposure time, rather than the default HST collecting area and 1 second exposures.
    #
    # If you really want to use the smaller test sample, you can use --sample test, but there
    # are only 100 galaxies there, so most galaxies will be repeated several times.

    if sample == 'test':
        cat = galsim.COSMOSCatalog('real_galaxy_catalog_23.5_example.fits', dir='data',
                                   area=roman.collecting_area, exptime=roman.exptime)
    else:
        cat = galsim.COSMOSCatalog(sample=sample, area=roman.collecting_area, exptime=roman.exptime)
    logger.info('Read in %d galaxies from catalog'%cat.nobjects)

    # Pick a plausible observation that might be made to celebrate Nancy Grace Roman's 100th
    # birthday.  (AG Draconis)
    ra_targ = galsim.Angle.from_hms('16:01:41.01257')
    dec_targ = galsim.Angle.from_dms('66:48:10.1312')
    targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)

    # Get the WCS for an observation at this position.
    # The date is NGR's 100th birthday.
    # Unfortunately, it now looks like Roman Space Telescope probably won't launch in time for
    # this commemerative observation.  Alas...
    date = datetime.datetime(2025, 5, 16)

    # We omit the position angle (PA) of the observatory, which means that it will just find the
    # optimal one (the one that has the solar panels pointed most directly towards the Sun given
    # this targ_pos and date).
    # The output of this routine is a dict of WCS objects, one for each SCA.  We then take the WCS
    # for the SCA that we are using.
    wcs_dict = roman.getWCS(world_pos=targ_pos, SCAs=use_SCA, date=date)
    wcs = wcs_dict[use_SCA]

    # Now start looping through the filters to draw.
    for ifilter, filter_name in enumerate(filters):

        logger.info('Beginning work for {0}.'.format(filter_name))

        # GalSim uses the term Bandpass for the class that defines the throughput across a
        # filter bandpass, partly because "filter" is a reserved word in python.  So we follow
        # that convention here as well.
        bandpass = roman_filters[filter_name]

        # Create the PSF
        # We are ignoring the position-dependence of the PSF within each SCA, just using the PSF
        # at the center of the sensor.
        # Note: pupil_bin=8 is faster at the expense of lower resolution for the diffraction spikes.
        # The n_waves keyword calculates this many PSF models in advance at different wavelengths
        # which we can interpolate between when drawing the galaxies.  For more accuracy w.r.t.
        # the chromaticity, you can increase this value of `n_waves`.
        # Note: Removing n_waves parameter would actually be both slower and less accurate, since
        # the OpticalPSF model would redo the wavefront calculation for each galaxy, and then
        # would still make an approximation that would be similar to n_waves=3.
        logger.info('Building PSF for SCA %d, filter %s.'%(use_SCA, filter_name))
        psf = roman.getPSF(use_SCA, filter_name, n_waves=10, wcs=wcs, pupil_bin=8)

        # Drawing PSF.  Note that the PSF object intrinsically has a flat SED, so if we convolve it
        # with a galaxy, it will properly take on the SED of the galaxy.  For the sake of this demo,
        # we will simply convolve with a 'star' that has a flat SED and unit flux in this band, so
        # that the PSF image will be normalized to unit flux. This does mean that the PSF image
        # being drawn here is not quite the right PSF for the galaxy.  Indeed, the PSF for the
        # galaxy effectively varies within it, since it differs for the bulge and the disk.  To make
        # a real image, one would have to choose SEDs for stars and convolve with a star that has a
        # reasonable SED, but we just draw with a flat SED for this demo.
        psf_filename = os.path.join(outpath, 'demo13_PSF_{0}.fits'.format(filter_name))

        # Generate a point source.
        point = galsim.DeltaFunction(flux=1.)

        # Use a flat SED here, but could use something else.  A stellar SED for instance.
        # Or a typical galaxy SED.  Depending on your purpose for drawing the PSF.
        star_sed = galsim.SED(lambda x:1, 'nm', 'flambda')

        # Give it unit flux in this filter.
        star_sed = star_sed.withFlux(1.,bandpass)

        # Convolve with the PSF
        star = galsim.Convolve(point*star_sed, psf)

        # Draw it.
        img_psf = galsim.ImageF(64,64)
        star.drawImage(bandpass=bandpass, image=img_psf, scale=roman.pixel_scale)
        img_psf.write(psf_filename)
        logger.debug('Wrote PSF {0}-band to {1}'.format(filter_name, psf_filename))

        # Set up the full image for the galaxies
        full_image = galsim.ImageF(roman.n_pix, roman.n_pix, wcs=wcs)

        # We have one rng for image-level stuff, and two others for the galaxies.
        # There are simpler ways to do this in a python script (e.g. probably only need 2
        # rngs, not 3), but this way of setting it up matches the way the config file initializes
        # the random number generators.
        image_rng = galsim.UniformDeviate(seed + ifilter * ngal)

        # Start with the flux from the sky. This is a little easier to do first before adding
        # the light from the galaxies, since we will have to apply Poisson noise to the sky flux
        # manually, but the photon shooting will automatically include Poisson noise for the
        # objects.

        # First we get the amount of zodaical light for a position corresponding to the center of
        # this SCA.  The results are provided in units of e-/arcsec^2, using the default Roman
        # exposure time since we did not explicitly specify one.  Then we multiply this by a factor
        # >1 to account for the amount of stray light that is expected.  If we do not provide a date
        # for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        # ecliptic coordinates) in 2025.
        SCA_cent_pos = wcs.toWorld(full_image.true_center)
        sky_level = roman.getSkyLevel(bandpass, world_pos=SCA_cent_pos)
        sky_level *= (1.0 + roman.stray_light_fraction)

        # Note that makeSkyImage() takes a bit of time. If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by roman.pixel_scale**2, and add that to full_image.
        wcs.makeSkyImage(full_image, sky_level)

        # The other background is the expected thermal backgrounds in this band.
        # These are provided in e-/pix/s, so we have to multiply by the exposure time.
        full_image += roman.thermal_backgrounds[filter_name]*roman.exptime

        # Save the current state as the expected value of the background, so we can subtract
        # it off later.
        sky_image = full_image.copy()

        # So far these are technically expectation values, so use Poisson noise to convert to
        # the realized number of electrons.
        poisson_noise = galsim.PoissonNoise(image_rng)
        full_image.addNoise(poisson_noise)

        # Draw the galaxies into the image.
        # We want (most of) the galaxy properties to be the same for all the filters.
        # E.g. the position, orintation, etc. should match up for all the observations.
        # To make this happen, we start an rng from the same seed each time.
        for i_gal in range(ngal):
            logger.info('Drawing image for object {} for band {}'.format(i_gal, filter_name))

            # The rng for galaxy parameters should be the same for each filter to make sure
            # we get the same parameters, position, rotation in each color.
            gal_rng = galsim.UniformDeviate(seed + 1 + 10**6 + i_gal)
            # The rng for photon shooting should be different for each filter.
            phot_rng = galsim.UniformDeviate(seed + 1 + i_gal + ifilter*ngal)

            # Pick a random position in the image to draw it.
            # If we had a real galaxy catalog with positions in terms of RA, Dec we could use
            # wcs.toImage() to find where those objects should be in terms of (x, y).
            # Note that we could use wcs.toWorld() to get the (RA, Dec) for these (x, y) positions
            # if we wanted that information, but we don't need it.
            x = gal_rng() * roman.n_pix
            y = gal_rng() * roman.n_pix
            image_pos = galsim.PositionD(x,y)

            # Select a random galaxy from the catalog.
            # If using the full COSMOS catalog downloaded via galsim_download_cosmos, you should
            # remove the weight=False option to enable the weighted selection available in the
            # full catalogs.
            gal = cat.makeGalaxy(chromatic=True, gal_type='parametric', rng=gal_rng)

            # Rotate the galaxy randomly
            theta = gal_rng() * 2 * np.pi * galsim.radians
            gal = gal.rotate(theta)

            # Convolve the (chromatic) galaxy with the (chromatic) PSF.
            final = galsim.Convolve(gal, psf)
            stamp = final.drawImage(bandpass, center=image_pos, wcs=wcs.local(image_pos),
                                    method='phot', rng=phot_rng)

            # Find the overlapping bounds between the large image and the individual stamp.
            bounds = stamp.bounds & full_image.bounds

            # Add this to the corresponding location in the large image.
            full_image[bounds] += stamp[bounds]

        # Now we're done with the per-galaxy drawing for this image.  The rest will be done for the
        # entire image at once.
        logger.info('Postage stamps of all galaxies drawn on a single big image for this filter.')
        logger.info('Adding the noise and detector non-idealities.')

        # Now that all sources of signal (from astronomical objects and background) have been added
        # to the image, we can start adding noise and detector effects.  There is a utility,
        # galsim.roman.allDetectorEffects(), that can apply ALL implemented noise and detector
        # effects in the proper order.  Here we step through the process and explain these in a bit
        # more detail without using that utility.

        # At this point in the image generation process, an integer number of photons gets
        # detected, unless any of the pre-noise values were > 2^30. That's when our Poisson
        # implementation switches over to the Gaussian approximation, which won't necessarily
        # produce integers.  This situation does not arise in practice for this demo, but if it did,
        # we could use full_image.quantize() to enforce integer pixel values.

        # The subsequent steps account for the non-ideality of the detectors.

        # 1) Reciprocity failure:
        # Reciprocity, in the context of photography, is the inverse relationship between the
        # incident flux (I) of a source object and the exposure time (t) required to produce a given
        # response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this relation does
        # not hold always. The pixel response to a high flux is larger than its response to a low
        # flux. This flux-dependent non-linearity is known as 'reciprocity failure', and the
        # approximate amount of reciprocity failure for the Roman detectors is known, so we can
        # include this detector effect in our images.

        # If we had wanted to, we could have specified a different exposure time than the default
        # one for Roman, but otherwise the following routine does not take any arguments.
        roman.addReciprocityFailure(full_image)
        logger.debug('Included reciprocity failure in {0}-band image'.format(filter_name))

        # 2) Adding dark current to the image:
        # Even when the detector is unexposed to any radiation, the electron-hole pairs that
        # are generated within the depletion region due to finite temperature are swept by the
        # high electric field at the junction of the photodiode. This small reverse bias
        # leakage current is referred to as 'dark current'. It is specified by the average
        # number of electrons reaching the detectors per unit time and has an associated
        # Poisson noise since it is a random event.
        dark_current = roman.dark_current*roman.exptime
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(image_rng, dark_current))
        full_image.addNoise(dark_noise)
        sky_image += dark_current # (also want to subtract this expectation value along with sky)

        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the
        # image generation process. We subtract these backgrounds in the end.

        # 3) Applying a quadratic non-linearity:
        # In order to convert the units from electrons to ADU, we must use the gain factor. The gain
        # has a weak dependency on the charge present in each pixel. This dependency is accounted
        # for by changing the pixel values (in electrons) and applying a constant nominal gain
        # later, which is unity in our demo.

        # Apply the Roman nonlinearity routine, which knows all about the nonlinearity expected in
        # the Roman detectors.
        roman.applyNonlinearity(full_image)

        # Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
        # detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
        # following syntax:
        # full_image.applyNonlinearity(NLfunc=NLfunc)
        # with NLfunc being a callable function that specifies how the output image pixel values
        # should relate to the input ones.
        logger.debug('Applied nonlinearity to {0}-band image'.format(filter_name))

        # 4) Including Interpixel capacitance:
        # The voltage read at a given pixel location is influenced by the charges present in the
        # neighboring pixel locations due to capacitive coupling of sense nodes. This interpixel
        # capacitance effect is modeled as a linear effect that is described as a convolution of a
        # 3x3 kernel with the image.  The Roman IPC routine knows about the kernel already, so the
        # user does not have to supply it.
        roman.applyIPC(full_image)
        logger.debug('Applied interpixel capacitance to {0}-band image'.format(filter_name))

        # 5) Adding read noise:
        # Read noise is the noise due to the on-chip amplifier that converts the charge into an
        # analog voltage.  We already applied the Poisson noise due to the sky level, so read noise
        # should just be added as Gaussian noise:
        read_noise = galsim.GaussianNoise(image_rng, sigma=roman.read_noise)
        full_image.addNoise(read_noise)
        logger.debug('Added readnoise to {0}-band image'.format(filter_name))

        # We divide by the gain to convert from e- to ADU. Currently, the gain value in the Roman
        # module is just set to 1, since we don't know what the exact gain will be, although it is
        # expected to be approximately 1. Eventually, this may change when the camera is assembled,
        # and there may be a different value for each SCA. For now, there is just a single number,
        # which is equal to 1.
        full_image /= roman.gain
        sky_image /= roman.gain

        # Finally, the analog-to-digital converter reads in an integer value.
        full_image.quantize()
        sky_image.quantize()
        # Note that the image type after this step is still a float.  If we want to actually
        # get integer values, we can do new_img = galsim.Image(full_image, dtype=int)

        # Since many people are used to viewing background-subtracted images, we provide a
        # version with the background subtracted (also rounding that to an int).
        full_image -= sky_image

        logger.debug('Subtracted background for {0}-band image'.format(filter_name))
        # Write the final image to a file.
        out_filename = os.path.join(outpath,'demo13_{0}.fits'.format(filter_name))
        full_image.write(out_filename)

        logger.info('Completed {0}-band image.'.format(filter_name))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -zoom 0.3 -scale limits -10 100 -rgb '+
                '-red output/demo13_H158.fits '+
                '-green output/demo13_J129.fits '+
                '-blue output/demo13_Y106.fits')

if __name__ == "__main__":
    main(sys.argv[1:])
