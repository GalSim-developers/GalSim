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

This script introduces the non-idealities arising from the (NIR) detectors, in particular those
that will be observed and accounted for in the WFIRST surveys. Four such non-ideal effects are
demonstrated, in the order in which they are introduced in the detectors:

1) Reciprocity Failure: Flux dependent sensitivity of the detector
2) Dark current: Constant response to zero flux, due to thermal generation of electron-hole pairs.
3) Non-linearity: Charge dependent gain in converting from units of electrons to ADU.
4) Interpixel Capacitance: Influence of charge in a pixel on the voltage reading of neighboring
   ones.

The purpose of the demo is two-fold: to show the effects of detector non-idealities in the full
context of the entire image generation process., including all sources of noise and skylevel added
at appropriate stages. After each effect, suggested parameters for viewing the intermediate and
difference images in ds9 are also included.

New feautres introduced in this demo:
- Adding sky level and dark current
- poisson_noise = galsim.PoissonNoise(rng)
- image.addReciprocityFailure(exp_time, alpha, base_flux)
- image.applyNonlinearity(NLfunc,*args)
- image.applyIPC(IPC_kernel, edge_treatment, fill_value, kernel_nonnegativity,
                 kernel_normalization)
- readnoise = galsim.CCDNoise(rng)
- readnoise.setReadNoise(readnoise_level)
"""

import numpy
import sys, os
import math
import logging
import time
import galsim as galsim
import galsim.wfirst as wfirst

def main(argv):
    # Increase the maximum fft size to draw galaxies convolved with PSF
    # NOTE TO SELF: NEEDS A FIX
    gsparams = galsim.GSParams()
    gsparams.maximum_fft_size = 8192
    print galsim.GSParams().maximum_fft_size

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
    logger.info('Poisson Noise model created')

    # Read in the WFIRST filters.
    filters = wfirst.getBandpasses(AB_zeropoint=True)
    # We care about the NIR imaging, not the prism and grism, so let's remove them:
    del filters['SNPrism']
    del filters['BAO-Grism']
    logger.debug('Read in filters')

    # Read in SEDs.  We only need two of them, for the two components of the galaxy (bulge and
    # disk).
    SED_names = ['CWW_E_ext', 'CWW_Im_ext']
    SEDs = {}
    for SED_name in SED_names:
        SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
        SED = galsim.SED(SED_filename, wave_type='Ang')
        
        # The normalization of SEDs affects how many photons are eventually drawn into an image.
        # One way to control this normalization is to specify the magnitude in a given bandpass
        # filter. We pick W149 and enforce the flux through the filter to be of magnitude specified
        # by `mag_norm`. This choice is overall normalization is completely arbitrary, but it
        # means that the colors of the galaxy will now be meaningful (for example, the bulge will
        # be more evident in the redder bands and the disk in the bluer bands).
        bandpass = filters['W149']
        mag_norm = 22.0

        SEDs[SED_name] = SED.withMagnitude(target_magnitude=mag_norm, bandpass=bandpass)

    logger.debug('Successfully read in SEDs')

    logger.info('')
    redshift = 0.8
    logger.info('Simulating chromatic bulge+disk galaxies from a catalog at z=%.1f'%redshift)

    bulge_SED = SEDs['CWW_E_ext'].atRedshift(redshift)
    disk_SED = SEDs['CWW_Im_ext'].atRedshift(redshift)

    logger.info('')
    logger.info('Reading from a catalog')
    #Read in a galaxy catalog
    cat_file_name = 'galsim_default_input.asc'
    dir = 'input'
    cat = galsim.Catalog(cat_file_name,dir=dir)
    logger.info('Read in %d galaxies from catalog',cat.nobjects)
    # Just use a few galaxies.
    n_use = 5

    # Here we carry out the initial steps that are necessary to get a fully chromatic PSF.  We use
    # the getPSF() routine in the WFIRST module, which knows all about the telescope parameters
    # (diameter, bandpasses, obscuration, etc.).  Note that we are going to arbitrarily choose a
    # single SCA rather than all of them, for faster calculations, and we're going to use a simpler
    # representation of the struts for faster calculations.  To do a more exact calculation of the
    # chromaticity and pupil plane configuration, remove the approximate_struts and the n_waves
    # keyword from this call:
    use_SCA = 7 # This could be any number from 1...18
    logger.info('Doing expensive pre-computation of PSF')
    t1 = time.time()
    PSFs = wfirst.getPSF(SCAs=use_SCA, approximate_struts=True, n_waves=10)
    PSF = PSFs[use_SCA]
    t2 = time.time()
    logger.info('Done PSF precomputation in %.1f seconds!'%(t2-t1))

    # Define some parameters
    stamp_size = 64

    # We are going to choose a particular location on the sky.
    ra_targ = 30.*galsim.degrees
    dec_targ = -10.*galsim.degrees
    targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)
    ang = 0.*galsim.degrees
    # Get the WCS for an observation at this position, with the focal plane array oriented at an
    # angle of 0 with respect to North.  The output of this routine is a list of WCS objects, one
    # for each SCA.  We will then take the WCS for the SCA that we are going to use.
    wcs_list = wfirst.getWCS(ang, pos=targ_pos, PA_is_FPA=True)
    wcs = wcs_list[use_SCA]
    # We need to find the center position for this SCA.  We'll tell it to give us a CelestialCoord
    # corresponding to (X, Y) = (wfirst.n_pix/2, wfirst.n_pix/2).
    SCA_cent_pos = wcs.toWorld(galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))

    # We are going to just randomly distribute points in X, Y.  If we had a real galaxy catalog
    # with positions in terms of RA, dec we could use wcs.toImage() to find where those objects
    # should be in terms of (X, Y).
    pos_rng = galsim.UniformDeviate(random_seed)
    # Make a list of (X, Y) values, eliminating the 10% of the edge pixels as the object centroids.
    x_stamp = []
    y_stamp = []
    theta_stamp = []
    for i_gal in xrange(n_use):
        x_stamp.append(pos_rng()*0.8*wfirst.n_pix + 0.1*wfirst.n_pix)
        y_stamp.append(pos_rng()*0.8*wfirst.n_pix + 0.1*wfirst.n_pix)
        theta_stamp.append(pos_rng()*2.0*numpy.pi*galsim.radians)

    # Drawing PSFs, defining the dict keys and recomputing skylevel for each filter
    for filter_name, filter_ in filters.iteritems():
        # Drawing PSF.  Note that the PSF object intrinsically has a flat SED, so if we
        # convolve it with a galaxy, it will properly take on the SED of the galaxy.  However,
        # this does mean that the PSF image being drawn here is not quite the right PSF for
        # the galaxy.  Indeed, the PSF for the galaxy effectively varies within it, since it
        # differs for the bulge and the disk.  However, the WFIRST bandpasses are narrow
        # enough that this doesn't matter too much.
        out_filename = os.path.join(outpath, 'demo13_PSF_{0}.fits'.format(filter_name))
        img_psf = galsim.ImageF(64,64)
        PSF.drawImage(bandpass=filter_, image=img_psf, scale=wfirst.pixel_scale)
        # Artificially normalize to a total flux of 1 for display purposes.
        img_psf /= img_psf.array.sum()
        img_psf.write(out_filename)
        logger.debug('Created PSF with flat SED for {0}-band'.format(filter_name))

        # First we get the amount of zodaical light for a position corresponding to the center of
        # this SCA.  Since we have supplied an exposure time, the results will be returned to us in
        # e-/s.  Then we multiply this by a factor to account for the amount of stray light that is
        # expected.
        sky_level_pix = wfirst.getSkyLevel(filters[filter_name], position=SCA_cent_pos,
                                           exp_time=wfirst.exptime)
        sky_level_pix *= (1.0 + wfirst.stray_light_fraction)
        # Finally we add the expected thermal backgrounds in this band.  These are provided in
        # e-/pix/s, so we have to multiply by the exposure time.
        sky_level_pix += wfirst.thermal_backgrounds[filter_name]*wfirst.exptime

        # Set up the final image:
        # final_image = galsim.ImageF(wfirst.n_pix,wfirst.n_pix, wcs=wcs)
        final_image = galsim.ImageF(wfirst.n_pix,wfirst.n_pix)

        for k in xrange(n_use): #xrange(cat.nobjects):
            logger.info('Processing the object at row %d in the input catalog'%k)

            # Galaxy is a bulge + disk with parameters taken from the catalog:
            disk = galsim.Exponential(half_light_radius=0.1*cat.getFloat(k,5))
            disk = disk * disk_SED
            disk = disk.shear(e1=cat.getFloat(k,6), e2=cat.getFloat(k,7))

            bulge = galsim.DeVaucouleurs(half_light_radius=0.1*cat.getFloat(k,8))
            bulge = bulge * bulge_SED
            bulge = bulge.shear(e1=cat.getFloat(k,9), e2=cat.getFloat(k,10))

            # Add the components to get the galaxy. The flux is clearly getting rescaled in an
            # incorrect way, which I need to fix, but for now we'll just artificially fix it.
            gal = 100*(1./3)*bulge+100*(2./3)*disk
            # At this stage, our galaxy is chromatic.
            logger.debug('Created bulge+disk galaxy final profile')
            # Q: Should the flux be adjusted or set mag_norm at a lower magnitude?????

            # Apply a random rotation
            gal = gal.rotate(theta_stamp[k])

            # Account for the fractional part of the position:
            ix = int(math.floor(x_stamp[k]+0.5))
            iy = int(math.floor(y_stamp[k]+0.5))
            offset = galsim.PositionD(x_stamp[k]-ix, y_stamp[k]-iy)

            # Create a nominal bounds for the postage stamp
            stamp_bounds = galsim.BoundsI(ix-0.5*stamp_size, ix+0.5*stamp_size-1, 
                                        iy-0.5*stamp_size, iy+0.5*stamp_size-1)

            # The center of the object is normally placed at the center of the postage stamp image.
            # You can change that with shift:
            # gal = gal.shift(dx=cat.getFloat(k,11), dy=cat.getFloat(k,12))
            # NOTE TO SELF: Disabled since galaxies look cropped - Arun

            # Convolve the chromatic galaxy and the chromatic PSF
            final = galsim.Convolve([gal,PSF],gsparams=gsparams)

            logger.debug('Preprocessing for galaxy %d completed.'%k)

            # Find the overlapping bounds:
            bounds = stamp_bounds & final_image.bounds

            # draw profile through WFIRST filters
            final.drawImage(filter_, image=final_image[bounds], offset=offset, add_to_image=True)

        logger.info('Postage stamps of all galaxies drawn on a single big image.')
        logger.info('Adding the skylevel, noise and detector non-idealities')

        # Now we're done with the per-galaxy drawing for this image.  The rest will be done for the
        # entire image at once.

        # Adding sky level to the image.  
        final_image += sky_level_pix

        # Adding Poisson Noise
        final_image.addNoise(poisson_noise)

        logger.debug('Creating {0}-band image.'.format(filter_name))

        # The subsequent steps account for the non-ideality of the detectors

        # Accounting Reciprocity Failure:
        # Reciprocity, in the context of photography, is the inverse relationship between the
        # incident flux (I) of a source object and the exposure time (t) required to produce a
        # given response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this
        # relation does not hold always. The pixel response to a high flux is larger than its
        # response to a low flux. This flux-dependent non-linearity is known as 'Reciprocity
        # Failure'.

        # Save the image before applying the transformation to see the difference
        final_image_1 = final_image.copy()

        final_image.addReciprocityFailure(exp_time=wfirst.exptime, alpha=wfirst.reciprocity_alpha,
            base_flux=1.0)
        logger.debug('Accounted for Reciprocity Failure in {0}-band image'.format(filter_name))
        final_image_2 = final_image.copy()

        diff = final_image_2-final_image_1

        out_filename = os.path.join(outpath,'demo13_RecipFail_{0}.fits'.format(filter_name))
        final_image_2.write(out_filename)
        out_filename = os.path.join(outpath,'demo13_diff_RecipFail_{0}.fits'.format(filter_name))
        diff.write(out_filename)

        # At this point in the image generation process, an integer number of photons gets
        # detected, hence we have to round the pixel values to integers:
        final_image.quantize()

        # Adding dark current to the image
        # Even when the detector is unexposed to any radiation, the electron-hole pairs that
        # are generated within the depletion region due to finite temperature are swept by the
        # high electric field at the junction of the photodiode. This small reverse bias
        # leakage current is referred to as 'Dark current'. It is specified by the average
        # number of electrons reaching the detectors per unit time and has an associated
        # Poisson noise since it's a random event.
        dark_img = galsim.ImageF(bounds=final_image.bounds,
            init_value=wfirst.dark_current*wfirst.exptime)
        dark_img.addNoise(poisson_noise)
        final_image += dark_img

        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the 
        # image generation process. We subtract these backgrounds in the end.

        # Applying a quadratic non-linearity
        # In order to convert the units from electrons to ADU, we must multiply the image by a
        # gain factor. The gain has a weak dependency on the charge present in each pixel. This
        # dependency is accounted for by changing the pixel values (in electrons) and applying
        # a constant nominal gain later, which is unity in our demo.

        # Save the image before applying the transformation to see the difference
        final_image_3 = final_image.copy()

        NLfunc = wfirst.NLfunc        # a quadratic non-linear function
        final_image.applyNonlinearity(NLfunc)
        logger.debug('Applied Nonlinearity to {0}-band image'.format(filter_name))
        final_image_4 = final_image.copy()

        diff = final_image_4-final_image_3

        out_filename = os.path.join(outpath,'demo13_NL_{0}.fits'.format(filter_name))
        final_image_4.write(out_filename)
        out_filename = os.path.join(outpath,'demo13_diff_NL_{0}.fits'.format(filter_name))
        diff.write(out_filename)

        # Adding Interpixel Capacitance
        # The voltage read at a given pixel location is influenced by the charges present in
        # the neighboring pixel locations due to capacitive coupling of sense nodes. This
        # interpixel capacitance effect is modelled as a linear effect that is described as a
        # convolution of a 3x3 kernel with the image. The WFIRST kernel is not normalized to
        # have the entries add to unity and hence must be normalized inside the routine.

        img.applyIPC(IPC_kernel=wfirst.ipc_kernel,edge_treatment='extend',
                     kernel_normalization=True)
        # Here, we use `edge_treatment='extend'`, which pads the image with zeros before
        # applying the kernel. The central part of the image is retained.
        logger.debug('Applied interpixel capacitance to {0}-band image'.format(filter_name))
        final_image_5 = final_image.copy()

        diff = final_image_5-final_image_4

        out_filename = os.path.join(outpath,'demo13_IPC_{0}.fits'.format(filter_name))
        final_image_5.write(out_filename)
        out_filename = os.path.join(outpath,'demo13_diff_IPC_{0}.fits'.format(filter_name))
        diff.write(out_filename)

        # Adding Read Noise
        read_noise = galsim.CCDNoise(rng)
        read_noise.setReadNoise(wfirst.read_noise)
        #final_image.addNoise(read_noise)
        final_image_6 = final_image.copy()

        logger.debug('Added Readnoise to {0}-band image'.format(filter_name))

        # Technically we have to apply the gain, dividing the signal in e- by the voltage gain
        # in e-/ADU to get a signal in ADU.  For WFIRST the gain is expected to be around 1,
        # so it doesn't really matter, but for completeness we include this step.
        final_image /= wfirst.gain

        # Finally, the analog-to-digital converter reads in integer value.
        final_image.quantize()

        # Note that the image type after this step is still a float.  If we want to actually
        # get integer values, we can do new_img = galsim.Image(img, dtype=int)

        # Since many people are used to viewing background-subtracted images, we provide a
        # version with the background subtracted (also rounding that to an int)
        tot_sky_level = (sky_level_pix + wfirst.dark_current*wfirst.exptime)/wfirst.gain
        tot_sky_level = numpy.round(tot_sky_level)

        final_image -= tot_sky_level

        logger.debug('Subtracted background for {0}-band image'.format(filter_name))

        out_filename = os.path.join(outpath,'demo13_{0}.fits'.format(filter_name))
        final_image.write(out_filename)

        logger.info('Created {0}-band image.'.format(filter_name))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output/demo13_J129.fits -green '
        +'-scale limits'+' -0.25 1.0 output/demo13_W149.fits -red -scale limits -0.25'
        +' 1.0 output/demo13_Z087.fits -zoom 2 &')

if __name__ == "__main__":
    main(sys.argv)
