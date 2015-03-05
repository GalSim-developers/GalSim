# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

This script currently doesn't have an equivalent demo*.yaml or demo*.json file.

This script introduces non-idealities arising from NIR detectors, in particular those that will be
observed and accounted for in the WFIRST surveys. Three such non-ideal effects are demonstrated, in
the order in which they are introduced in the detectors:

1) Reciprocity failure: Flux-dependent sensitivity of the detector.
2) Non-linearity: Charge-dependent gain in converting from units of electrons to ADU.  Non-linearity
   in some form is also relevant for CCDs in addition to NIR detectors.
3) Interpixel capacitance: Influence of charge in a pixel on the voltage reading of neighboring
   ones.

The purpose of the demo is two-fold: (1) to show the effects of detector non-idealities on images
from NIR detectors, and (2) to illustrate the full image generation process, including all sources
of noise at appropriate stages.

New features introduced in this demo:
- galsim.makeCOSMOSCatalog(...)
- galsim.makeCOSMOSObj(...)
- image.quantize()
- Routines to include WFIRST-specific detector effects:
  - galsim.wfirst.addReciprocityFailure(image)
  - galsim.wfirst.applyNonlinearity(image)
  - galsim.wfirst.applyIPC(image)
- Routines to get basic information about WFIRST bandpasses, PSFs, and WCS:
  - galsim.wfirst.getBandpasses()
  - galsim.wfirst.getPSF()
  - galsim.wfirst.getWCS()
  - galsim.wfirst.getSkyLevel()
"""

import numpy
import sys, os
import math
import logging
import time
import galsim as galsim
import galsim.wfirst as wfirst

def main(argv):
    # Where to find and output data.
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "data/"))
    outpath = os.path.abspath(os.path.join(path, "output/"))

    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo13")

    # Initialize (pseudo-)random number generator.
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)

    # Generate a Poisson noise model.
    poisson_noise = galsim.PoissonNoise(rng) 
    logger.info('Poisson noise model created.')

    # Read in the WFIRST filters, setting an AB zeropoint appropriate for this telescope given its
    # diameter and (since we didn't use any keyword arguments to modify this) using the typical
    # exposure time for WFIRST images.
    filters = wfirst.getBandpasses(AB_zeropoint=True)
    logger.debug('Read in WFIRST imaging filters.')

    logger.info('Reading from a parametric COSMOS catalog.')
    # Read in a galaxy catalog - just a random subsample of 500 galaxies for F814W<23.5 from COSMOS.
    cat_file_name = 'real_galaxy_23.5_500_fits.fits'
    dir = 'data'
    # Use the routine that can take COSMOS real or parametric galaxy information, and tell it we
    # want parametric galaxies.
    cat = galsim.makeCOSMOSCatalog(cat_file_name, dir=dir, use_real=False)
    logger.info('Read in %d galaxies from catalog'%len(cat))
    # Just use a few galaxies, to save time.  Note that we are going to put 4000 galaxy images into
    # our big image, so if we have n_use=10, each galaxy will appear 400 times.  Users who want a
    # more interesting image with greater variation in the galaxy population can change `n_use` to
    # something larger that evenly divides 4000 (but it should be <=500, which is the number of
    # galaxies in the catalog).  The code is not set up to handle values of n_use that do not evenly
    # divide n_tot, though it would not be very hard to do so.  With 4000 galaxies in a 4k x 4k
    # image with the WFIRST pixel scale, the effective galaxy number density is 74/arcmin^2
    n_use = 10
    n_tot = 4000
    n_per_gal = n_tot / n_use

    # Here we carry out the initial steps that are necessary to get a fully chromatic PSF.  We use
    # the getPSF() routine in the WFIRST module, which knows all about the telescope parameters
    # (diameter, bandpasses, obscuration, etc.).  Note that we arbitrarily choose a single SCA
    # (Sensor Chip Assembly) rather than all of them, for faster calculations, and use a simple
    # representation of the struts for faster calculations.  To do a more exact calculation of the
    # chromaticity and pupil plane configuration, remove the `approximate_struts` and the `n_waves`
    # keyword from the call to getPSF():
    use_SCA = 7 # This could be any number from 1...18
    logger.info('Doing expensive pre-computation of PSF.')
    t1 = time.time()
    logger.setLevel(logging.DEBUG)
    PSFs = wfirst.getPSF(SCAs=use_SCA, approximate_struts=True, n_waves=10, logger=logger)
    logger.setLevel(logging.INFO)
    PSF = PSFs[use_SCA]
    t2 = time.time()
    logger.info('Done PSF precomputation in %.1f seconds!'%(t2-t1))

    # Define the size of the postage stamp that we use for each individual galaxy within the larger
    # image, and for the PSF images.
    stamp_size = 128

    # We choose a particular (RA, dec) location on the sky for our observation.
    ra_targ = 30.*galsim.degrees
    dec_targ = -10.*galsim.degrees
    targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)
    ang = 0.*galsim.degrees
    # Get the WCS for an observation at this position, with the focal plane array oriented at an
    # angle of `ang` with respect to North.  The output of this routine is a list of WCS objects,
    # one for each SCA.  We then take the WCS for the SCA that we are using.
    wcs_list = wfirst.getWCS(ang, world_pos=targ_pos, PA_is_FPA=True, SCAs=use_SCA)
    wcs = wcs_list[use_SCA]
    # We need to find the center position for this SCA.  We'll tell it to give us a CelestialCoord
    # corresponding to (X, Y) = (wfirst.n_pix/2, wfirst.n_pix/2).
    SCA_cent_pos = wcs.toWorld(galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))

    # We randomly distribute points in (X, Y) on the CCD.
    # If we had a real galaxy catalog with positions in terms of RA, dec we could use wcs.toImage()
    # to find where those objects should be in terms of (X, Y).
    pos_rng = galsim.UniformDeviate(random_seed)
    # Make a list of (X, Y) values.
    x_stamp = []
    y_stamp = []
    for i_gal in xrange(n_tot):
        x_stamp.append(pos_rng()*wfirst.n_pix)
        y_stamp.append(pos_rng()*wfirst.n_pix)
        # Note that we could use wcs.toWorld() to get the (RA, dec) for these (x, y) positions.  Or,
        # if we had started with (RA, dec) positions, we could have used wcs.toImage() to get the
        # CCD coordinates for those positions.

    # Make the 2-component parametric GSObjects for each object, including chromaticity (roughly
    # appropriate SEDs per galaxy component, at the appropriate galaxy redshift).  Note that since
    # the PSF is position-independent within the SCA, we can simply do the convolution with that PSF
    # now instead of using a different one for each position.
    gal_list = []
    for i_gal in xrange(n_use):
        logger.info('Processing the object at row %d in the input catalog.'%i_gal)

        # Use built-in routines for constructing COSMOS objects:
        gal = galsim.makeCOSMOSObj(cat, i_gal, chromatic=True)

        # Convolve the chromatic galaxy and the chromatic PSF
        final = galsim.Convolve(gal, PSF)
        logger.debug('Pre-processing for galaxy %d completed.'%i_gal)
        gal_list.append(final)

    # Calculate the sky level for each filter, and draw the PSF and the galaxies through the
    # filters.
    for filter_name, filter_ in filters.iteritems():
        logger.info('Beginning work for {0}.'.format(filter_name))

        # Drawing PSF.  Note that the PSF object intrinsically has a flat SED, so if we
        # convolve it with a galaxy, it will properly take on the SED of the galaxy.  However,
        # this does mean that the PSF image being drawn here is not quite the right PSF for
        # the galaxy.  Indeed, the PSF for the galaxy effectively varies within it, since it
        # differs for the bulge and the disk.  To make a real image, one would have to choose SEDs
        # for stars and convolve with a star that has a reasonable SED, but we just 
        # draw with a flat SED for this demo.
        out_filename = os.path.join(outpath, 'demo13_PSF_{0}.fits'.format(filter_name))
        img_psf = galsim.ImageF(64,64)
        PSF.drawImage(bandpass=filter_, image=img_psf, scale=wfirst.pixel_scale)
        # Artificially normalize to a total flux of 1 for display purposes.
        img_psf /= img_psf.array.sum()
        img_psf.write(out_filename)
        logger.debug('Created PSF with flat SED for {0}-band'.format(filter_name))

        # Set up the full image that will contain all the individual galaxy images, with information
        # about WCS:
        final_image = galsim.ImageF(wfirst.n_pix,wfirst.n_pix, wcs=wcs)

        # Draw the galaxies into the image.
        for i_gal in xrange(n_use):
            logger.info('Drawing image for the object at row %d in the input catalog'%i_gal)

            # We want to only draw the galaxy once (for speed), not over and over with different
            # sub-pixel offsets.  For this reason we ignore the sub-pixel offset entirely.
            stamp = galsim.Image(stamp_size, stamp_size)
            gal_list[i_gal].drawImage(filter_, image=stamp)

            # Have to find where to place it:
            for i_gal_use in range(i_gal*n_per_gal, i_gal*(n_per_gal+1)):
                # Account for the fractional part of the position:
                ix = int(math.floor(x_stamp[i_gal_use]+0.5))
                iy = int(math.floor(y_stamp[i_gal_use]+0.5))
                # We don't actually use this offset.
                offset = galsim.PositionD(x_stamp[i_gal]-ix, y_stamp[i_gal]-iy)

                # Create a nominal bound for the postage stamp given the integer part of the
                # position.
                stamp_bounds = galsim.BoundsI(ix-0.5*stamp_size, ix+0.5*stamp_size-1, 
                                              iy-0.5*stamp_size, iy+0.5*stamp_size-1)
                stamp.setOrigin(galsim.PositionI(stamp_bounds.xmin, stamp_bounds.ymin))

                # Find the overlapping bounds between the large image and the individual postage
                # stamp.
                bounds = stamp_bounds & final_image.bounds

                # Copy the image into the right place in the big image.  There is a difference in
                # normalization between COSMOS images and these that we account for using a
                # numerical factor.
                final_image[bounds] += stamp[bounds]/0.03**2

        # Now we're done with the per-galaxy drawing for this image.  The rest will be done for the
        # entire image at once.
        logger.info('Postage stamps of all galaxies drawn on a single big image for this filter.')
        logger.info('Adding the sky level, noise and detector non-idealities.')

        # First we get the amount of zodaical light for a position corresponding to the center of
        # this SCA.  The results are provided in units of e-/arcsec^2, using the default WFIRST
        # exposure time since we did not explicitly specify one.  Then we multiply this by a factor
        # >1 to account for the amount of stray light that is expected.  Technically one should make
        # a position-dependent sky level using wcs.makeSkyImage() to account for variable pixel
        # area, but this is a fairly flat function of position, so using a constant is not too
        # bad.
        sky_level_pix = wfirst.getSkyLevel(filters[filter_name], world_pos=SCA_cent_pos)
        sky_level_pix *= (1.0 + wfirst.stray_light_fraction)
        sky_level_pix *= wfirst.pixel_scale**2 # convert to e-/pix
        # Finally we add the expected thermal backgrounds in this band.  These are provided in
        # e-/pix/s, so we have to multiply by the exposure time.
        sky_level_pix += wfirst.thermal_backgrounds[filter_name]*wfirst.exptime
        # Adding sky level to the image.  
        final_image += sky_level_pix

        # Now that all sources of signal (from astronomical objects and background) have been added,
        # we can include the expected Poisson noise:
        final_image.addNoise(poisson_noise)

        # The subsequent steps account for the non-ideality of the detectors.

        # 1) Reciprocity failure:
        # Reciprocity, in the context of photography, is the inverse relationship between the
        # incident flux (I) of a source object and the exposure time (t) required to produce a
        # given response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this
        # relation does not hold always. The pixel response to a high flux is larger than its
        # response to a low flux. This flux-dependent non-linearity is known as 'reciprocity
        # failure', and the approximate amount of reciprocity failure for the WFIRST detectors is
        # known, so we can include this detector effect in our images.

        # Save the image before applying the transformation to see the difference
        final_image_orig = final_image.copy()
        # If we had wanted to, we could have specified a different exposure time than the default
        # one for WFIRST, but otherwise the following routine does not take any arguments.
        wfirst.addReciprocityFailure(final_image)
        logger.debug('Included reciprocity failure in {0}-band image'.format(filter_name))
        final_image_2 = final_image.copy()
        # Isolate the changes due to reciprocity failure.
        diff = final_image_2-final_image_orig

        out_filename = os.path.join(outpath,'demo13_RecipFail_{0}.fits'.format(filter_name))
        final_image_2.write(out_filename)
        out_filename = os.path.join(outpath,'demo13_diff_RecipFail_{0}.fits'.format(filter_name))
        diff.write(out_filename)

        # At this point in the image generation process, an integer number of photons gets
        # detected, hence we have to round the pixel values to integers:
        final_image.quantize()

        # 2) Adding dark current to the image:
        # Even when the detector is unexposed to any radiation, the electron-hole pairs that
        # are generated within the depletion region due to finite temperature are swept by the
        # high electric field at the junction of the photodiode. This small reverse bias
        # leakage current is referred to as 'Dark current'. It is specified by the average
        # number of electrons reaching the detectors per unit time and has an associated
        # Poisson noise since it is a random event.
        dark_img = galsim.ImageF(bounds=final_image.bounds,
                                 init_value=wfirst.dark_current*wfirst.exptime)
        dark_img.addNoise(poisson_noise)
        final_image += dark_img

        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the 
        # image generation process. We subtract these backgrounds in the end.

        # 3) Applying a quadratic non-linearity:
        # In order to convert the units from electrons to ADU, we must multiply the image by a
        # gain factor. The gain has a weak dependency on the charge present in each pixel. This
        # dependency is accounted for by changing the pixel values (in electrons) and applying
        # a constant nominal gain later, which is unity in our demo.

        # Save the image before applying the transformation to see the difference:
        final_image_3 = final_image.copy()

        # Apply the WFIRST nonlinearity routine, which knows all about the nonlinearity expected in
        # the WFIRST detectors.
        wfirst.applyNonlinearity(final_image)
        # Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
        # detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
        # following syntax:
        # final_image.applyNonlinearity(NLfunc=NLfunc)
        # with NLfunc being a callable function that specifies how the output image pixel values
        # should relate to the input ones.
        logger.debug('Applied nonlinearity to {0}-band image'.format(filter_name))
        final_image_4 = final_image.copy()
        # Isolate the changes due to non-linear gain.
        diff = final_image_4-final_image_3

        out_filename = os.path.join(outpath,'demo13_NL_{0}.fits'.format(filter_name))
        final_image_4.write(out_filename)
        out_filename = os.path.join(outpath,'demo13_diff_NL_{0}.fits'.format(filter_name))
        diff.write(out_filename)

        # 4) Including Interpixel capacitance:
        # The voltage read at a given pixel location is influenced by the charges present in
        # the neighboring pixel locations due to capacitive coupling of sense nodes. This
        # interpixel capacitance effect is modeled as a linear effect that is described as a
        # convolution of a 3x3 kernel with the image.  The WFIRST IPC routine knows about the kernel
        # already, so the user does not have to supply it.
        wfirst.applyIPC(final_image)
        # Here, we use `edge_treatment='extend'`, which pads the image with zeros before
        # applying the kernel. The central part of the image is retained.
        logger.debug('Applied interpixel capacitance to {0}-band image'.format(filter_name))
        final_image_5 = final_image.copy()
        # Isolate the changes due to the interpixel capacitance effect.
        diff = final_image_5-final_image_4

        out_filename = os.path.join(outpath,'demo13_IPC_{0}.fits'.format(filter_name))
        final_image_5.write(out_filename)
        out_filename = os.path.join(outpath,'demo13_diff_IPC_{0}.fits'.format(filter_name))
        diff.write(out_filename)

        # 5) Adding read noise:
        # Read noise is the noise due to the on-chip amplifier that converts the charge into an
        # analog voltage.
        read_noise = galsim.CCDNoise(rng)
        read_noise.setReadNoise(wfirst.read_noise)
        final_image.addNoise(read_noise)
        final_image_6 = final_image.copy()

        logger.debug('Added readnoise to {0}-band image'.format(filter_name))

        # Technically we have to apply the gain, dividing the signal in e- by the voltage gain
        # in e-/ADU to get a signal in ADU.  For WFIRST the gain is expected to be around 1,
        # so it doesn't really matter, but for completeness we include this step.
        final_image /= wfirst.gain

        # Finally, the analog-to-digital converter reads in an integer value.
        final_image.quantize()
        # Note that the image type after this step is still a float.  If we want to actually
        # get integer values, we can do new_img = galsim.Image(final_image, dtype=int)

        # Since many people are used to viewing background-subtracted images, we provide a
        # version with the background subtracted (also rounding that to an int).
        tot_sky_level = (sky_level_pix + wfirst.dark_current*wfirst.exptime)/wfirst.gain
        tot_sky_level = numpy.round(tot_sky_level)
        final_image -= tot_sky_level

        logger.debug('Subtracted background for {0}-band image'.format(filter_name))
        # Write the final image to a file.
        out_filename = os.path.join(outpath,'demo13_{0}.fits'.format(filter_name))
        final_image.write(out_filename)

        logger.info('Completed {0}-band image.'.format(filter_name))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output/demo13_J129.fits -green '
        +'-scale limits'+' -0.25 1.0 output/demo13_W149.fits -red -scale limits -0.25'
        +' 1.0 output/demo13_Z087.fits -zoom 2 &')

if __name__ == "__main__":
    main(sys.argv)
