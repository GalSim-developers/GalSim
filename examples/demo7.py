"""
Demo #7

The seventh script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script introduces drawing profiles with photon shooting rather than doing the 
convolution with an FFT.  It makes images using 5 different kinds of PSF and 5 different
kinds of galaxy.  Some of the parameters (flux, size and shape) are random variables, so 
each of the 25 pairings is drawn 4 times with different realizations of the random numbers.
The profiles are drawn twice, once with the FFT method, and once with photon shooting.
The two images are drawn side by side so it is easy to visually compare the results.
The 100 total profiles are written to a FITS data cube, which makes it easy to scroll
through the images comparing the two drawing methods.


New features introduced in this demo:

- obj = galsim.Airy(lam_over_diam)
- obj2 = obj.copy()
- obj.applyDilation(scale)
- image.setScale(pixel_scale)
- obj.draw(image)  -- i.e. taking the scale from the image rather than a dx= argument
- obj.drawShoot(image, max_extra_noise, uniform_deviate)
- noise = galsim.PoissonDeviate(rng, mean)
"""

import sys
import os
import math
import numpy
import logging
import time
import galsim


def main(argv):
    """
    Make a fits image cube where each frame has two images of the same galaxy drawn 
    with regular FFT convolution and with photon shooting.

    We do this for 5 different PSFs and 5 different galaxies, each with 4 different (random)
    fluxes, sizes, and shapes.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo7")

    # To turn off logging:
    #logger.propagate = False

    # Define some parameters we'll use below.

    file_name = os.path.join('output','cube_phot.fits')

    random_seed = 1512413
    sky_level = 1.e4        # ADU / arcsec^2
    pixel_scale = 0.28      # arcsec
    nx = 64
    ny = 64

    gal_flux_min = 1.e4     # Range for galaxy flux
    gal_flux_max = 1.e5  
    gal_hlr_min = 0.3       # arcsec
    gal_hlr_max = 1.3       # arcsec
    gal_e_min = 0.          # Range for ellipticity
    gal_e_max = 0.8

    psf_fwhm = 0.65         # arcsec

    logger.info('Starting demo script 7')

    # Make the pixel:
    pix = galsim.Pixel(xw = pixel_scale)

    # Make the PSF profiles:
    psf1 = galsim.Gaussian(fwhm = psf_fwhm)
    psf2 = galsim.Moffat(fwhm = psf_fwhm, beta = 2.4)
    psf3_inner = galsim.Gaussian(fwhm = psf_fwhm, flux = 0.8)
    psf3_outer = galsim.Gaussian(fwhm = 2*psf_fwhm, flux = 0.2)
    psf3 = psf3_inner + psf3_outer
    atmos = galsim.Gaussian(fwhm = psf_fwhm)
    optics = galsim.OpticalPSF(
            lam_over_diam = 0.6 * psf_fwhm,
            obscuration = 0.4,
            defocus = 0.1,
            astig1 = 0.3, astig2 = -0.2,
            coma1 = 0.2, coma2 = 0.1,
            spher = -0.3) 
    psf4 = galsim.Convolve([atmos,optics])
    #atmos = galsim.AtmosphericPSF(fwhm = psf_fwhm)
    atmos = galsim.Kolmogorov(fwhm = psf_fwhm)
    optics = galsim.Airy(lam_over_diam = 0.3 * psf_fwhm) 
    psf5 = galsim.Convolve([atmos,optics])
    psfs = [psf1, psf2, psf3, psf4, psf5]
    psf_names = ["Gaussian", "Moffat", "Double Gaussian", "OpticalPSF", "Kolmogorov * Airy"]
    psf_times = [0,0,0,0,0]
    psf_fft_times = [0,0,0,0,0]
    psf_phot_times = [0,0,0,0,0]

    # Make the galaxy profiles:
    gal1 = galsim.Gaussian(half_light_radius = 1)
    gal2 = galsim.Exponential(half_light_radius = 1)
    gal3 = galsim.DeVaucouleurs(half_light_radius = 1)
    gal4 = galsim.Sersic(half_light_radius = 1, n = 2.5)
    bulge = galsim.Sersic(half_light_radius = 0.7, n = 3.2)
    disk = galsim.Sersic(half_light_radius = 1.2, n = 1.5)
    gal5 = 0.4*bulge + 0.6*disk  # Net half-light radius is only approximate for this one.
    gals = [gal1, gal2, gal3, gal4, gal5]
    gal_names = ["Gaussian", "Exponential", "Devaucouleurs", "n=2.5 Sersic", "Bulge + Disk"]
    gal_times = [0,0,0,0,0]
    gal_fft_times = [0,0,0,0,0]
    gal_phot_times = [0,0,0,0,0]

    # Other times to keep track of:
    setup_times = 0
    fft_times = 0
    phot_times = 0
    noise_times = 0

    # Loop over combinations of psf, gal, and make 4 random choices for flux, size, shape.
    all_images = []
    k = 0
    for ipsf in range(len(psfs)):
        psf = psfs[ipsf]
        psf_name = psf_names[ipsf]
        for igal in range(len(gals)):
            gal = gals[igal]
            gal_name = gal_names[igal]
            for i in range(4):
                logger.debug('Start work on image %d',i)
                t1 = time.time()

                # Initialize the random number generator we will be using.
                rng = galsim.UniformDeviate(random_seed+k)

                # Get an new copy, so we'll want to keep the original unmodified.
                gal1 = gal.copy()

                # Generate random variates:
                flux = rng() * (gal_flux_max-gal_flux_min) + gal_flux_min
                gal1.setFlux(flux)

                hlr = rng() * (gal_hlr_max-gal_hlr_min) + gal_hlr_min
                gal1.applyDilation(hlr)

                beta_ellip = rng() * 2*math.pi * galsim.radians
                ellip = rng() * (gal_e_max-gal_e_min) + gal_e_min
                gal_shape = galsim.Shear(e=ellip, beta=beta_ellip)
                gal1.applyShear(gal_shape)

                # Build the final object by convolving the galaxy, PSF and pixel response.
                final = galsim.Convolve([psf, pix, gal1])
                # For photon shooting, need a version without the pixel (see below).
                final_nopix = galsim.Convolve([psf, gal1])

                # Create the large, double width output image
                image = galsim.ImageF(2*nx+2,ny)

                # Rather than provide a dx= argument to the draw commands, we can also
                # set the pixel scale in the image itself with setScale.
                image.setScale(pixel_scale)

                # Assign the following two "ImageViews", fft_image and phot_image.
                # Using the syntax below, these are views into the larger image.  
                # Changes/additions to the sub-images referenced by the views are automatically 
                # reflected in the original image.
                fft_image = image[galsim.BoundsI(1, nx, 1, ny)]
                phot_image = image[galsim.BoundsI(nx+3, 2*nx+2, 1, ny)]

                logger.debug('   Read in training sample galaxy and PSF from file')
                t2 = time.time()

                # Draw the profile
                final.draw(fft_image)

                logger.debug('   Drew fft image.  Total drawn flux = %f.  .flux = %f',
                        fft_image.array.sum(),final.getFlux())
                t3 = time.time()

                # Add Poisson noise
                sky_level_pixel = sky_level * pixel_scale**2
                fft_image += sky_level_pixel
                fft_image.addNoise(galsim.CCDNoise(rng))
                fft_image -= sky_level_pixel

                t4 = time.time()

                # The next two lines are just to get the output from this demo script
                # to match the output from the parsing of demo7.yaml.
                rng = galsim.UniformDeviate(random_seed+k)
                rng(); rng(); rng(); rng();

                # Repeat for photon shooting image.
                # Photon shooting automatically convolves by the pixel, so we've made sure not
                # to include it in the profile!
                final_nopix.drawShoot(phot_image, max_extra_noise=sky_level_pixel/100, 
                                      uniform_deviate=rng)
                t5 = time.time()

                # For photon shooting, galaxy already has poisson noise, so we want to make 
                # sure not to add that noise again!  Thus, we just add sky noise, which 
                # is Poisson with the mean = sky_level_pixel
                # Note: this won't add the mean level.  The effect on the pixels has an
                # expectation of 0 -- it just adds noise commensurate with the given mean.
                phot_image.addNoise(galsim.PoissonDeviate(rng, mean=sky_level_pixel))

                logger.debug('   Added Poisson noise.  Image fluxes are now %f and %f',
                        fft_image.array.sum(),phot_image.array.sum())
                t6 = time.time()

                # Store that into the list of all images
                all_images += [image]

                k = k+1
                logger.info('%d: %s * %s, flux = %.2e, hlr = %.2f, ellip = (%.2f,%.2f)',
                        k,gal_name, psf_name, flux, hlr, gal_shape.getE1(), gal_shape.getE2())
                logger.debug('   Times: %f, %f, %f, %f, %f',t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)

                psf_times[ipsf] += t6-t1
                psf_fft_times[ipsf] += t3-t2
                psf_phot_times[ipsf] += t5-t4
                gal_times[igal] += t6-t1
                gal_fft_times[igal] += t3-t2
                gal_phot_times[igal] += t5-t4
                setup_times += t2-t1
                fft_times += t3-t2
                phot_times += t5-t4
                noise_times += t4-t3 + t6-t5

    logger.info('Done making images of galaxies')
    logger.info('')
    logger.info('Some timing statistics:')
    logger.info('   Total time for setup steps = %f',setup_times)
    logger.info('   Total time for regular fft drawing = %f',fft_times)
    logger.info('   Total time for photon shooting = %f',phot_times)
    logger.info('   Total time for adding noise = %f',noise_times)
    logger.info('')
    logger.info('Breakdown by PSF type:')
    for ipsf in range(len(psfs)):
        logger.info('   %s: Total time = %f  (fft: %f, phot: %f)',
            psf_names[ipsf],psf_times[ipsf],psf_fft_times[ipsf],psf_phot_times[ipsf])
    logger.info('')
    logger.info('Breakdown by Galaxy type:')
    for igal in range(len(gals)):
        logger.info('   %s: Total time = %f  (fft: %f, phot: %f)',
            gal_names[igal],gal_times[igal],gal_fft_times[igal],gal_phot_times[igal])
    logger.info('')

    # Now write the image to disk.
    galsim.fits.writeCube(all_images, file_name, clobber=True)
    logger.info('Wrote fft image to fits data cube %r',file_name)


if __name__ == "__main__":
    main(sys.argv)
