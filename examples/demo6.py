#!/usr/bin/env python

"""
Some example scripts to make multi-object images using the GalSim library.
"""

import sys
import os
import math
import numpy
import logging
import time
import galsim

# Simulations with real galaxies from a catalog
def main(argv):
    """
    Make a fits image cube using real COSMOS galaxies from a catalog describing the training
    sample.

      - The number of images in the cube matches the number of rows in the catalog.
      - Each image size is computed automatically by GalSim based on the Nyquist size.
      - Both galaxies and stars.
      - PSF is a double Gaussian, the same for each galaxy.
      - Galaxies are randomly rotated to remove the imprint of any lensing shears in the COSMOS
        data.
      - The same shear is applied to each galaxy.
      - Noise is poisson using a nominal sky value of 1.e6
        the noise in the original COSMOS data.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo6")

    # Define some parameters we'll use below.

    cat_file_name = 'real_galaxy_catalog_example.fits'
    image_dir = 'data'
    cube_file_name = os.path.join('output','cube_real.fits')
    psf_file_name = os.path.join('output','psf_real.fits')

    random_seed = 1512413
    sky_level = 1.e6        # ADU / arcsec^2
    pixel_scale = 0.15      # arcsec
    gal_flux = 1.e5         # arbitrary choice, makes nice (not too) noisy images
    gal_g1 = -0.027         #
    gal_g2 = 0.031          #
    psf_inner_fwhm = 0.6    # arcsec
    psf_outer_fwhm = 2.3    # arcsec
    psf_inner_fraction = 0.8  # fraction of total PSF flux in the inner Gaussian
    psf_outer_fraction = 0.2  # fraction of total PSF flux in the inner Gaussian
    ngal = 100  

    logger.info('Starting demo script 6 using:')
    logger.info('    - real galaxies from catalog %r',cat_file_name)
    logger.info('    - double Gaussian PSF')
    logger.info('    - pixel scale = %.2f',pixel_scale)
    logger.info('    - Applied gravitational shear = (%.3f,%.3f)',gal_g1,gal_g2)
    logger.info('    - Poisson noise (sky level = %.1e).', sky_level)
    
    # Read in galaxy catalog
    real_galaxy_catalog = galsim.RealGalaxyCatalog(cat_file_name, image_dir)
    real_galaxy_catalog.preload()
    logger.info('Read in %d real galaxies from catalog', real_galaxy_catalog.nobjects)

    ## Make the ePSF
    # first make the double Gaussian PSF
    psf1 = galsim.Gaussian(fwhm = psf_inner_fwhm, flux = psf_inner_fraction)
    psf2 = galsim.Gaussian(fwhm = psf_outer_fwhm, flux = psf_outer_fraction)
    psf = psf1+psf2
    # make the pixel response
    pix = galsim.Pixel(xw = pixel_scale, yw = pixel_scale)
    # convolve PSF and pixel response function to get the effective PSF (ePSF)
    epsf = galsim.Convolve([psf, pix])
    # Draw this one with no noise.
    epsf_image = epsf.draw(dx = pixel_scale)
    # write to file
    epsf_image.write(psf_file_name, clobber = True)
    logger.info('Created ePSF and wrote to file %r',psf_file_name)

    # Build the images
    all_images = []
    for k in range(ngal):
        #logger.info('Start work on image %d',i)
        t1 = time.time()

        # Initialize the random number generator we will be using.
        rng = galsim.UniformDeviate(random_seed+k)

        gal = galsim.RealGalaxy(real_galaxy_catalog, index = k)
        #logger.info('   Read in training sample galaxy and PSF from file')
        t2 = time.time()

        # Set the flux
        gal.setFlux(gal_flux)

        # Rotate by a random angle
        theta = 2.*math.pi * rng() * galsim.radians
        gal.applyRotation(theta)

        # Apply the desired shear
        gal.applyShear(g1=gal_g1, g2=gal_g2)
        
        # Make the combined profile
        final = galsim.Convolve([psf, pix, gal])

        # Draw the profile
        if k == 0:
            im = final.draw(dx=pixel_scale)
            xsize, ysize = im.array.shape
        else:
            im = galsim.ImageF(xsize,ysize)
            final.draw(im, dx=pixel_scale)

        #logger.info('   Drew image')
        t3 = time.time()

        # Add Poisson noise
        sky_level_pixel = sky_level * pixel_scale**2
        im += sky_level_pixel
        # Again, default is gain=1, read_noise=0
        # So this is just poisson noise with the current image flux values.
        im.addNoise(galsim.CCDNoise(rng)) 
        im -= sky_level_pixel

        #logger.info('   Added Poisson noise')
        t4 = time.time()

        # Store that into the list of all images
        all_images += [im]
        t5 = time.time()

        #logger.info('   Times: %f, %f, %f, %f',t2-t1, t3-t2, t4-t3, t5-t4)
        logger.info('Image %d: size = %d x %d, total time = %f sec', k, xsize, ysize, t5-t1)

    logger.info('Done making images of galaxies')

    # Now write the image to disk.
    # We write the images to a fits data cube.
    galsim.fits.writeCube(all_images, cube_file_name, clobber=True)
    logger.info('Wrote image to fits data cube %r',cube_file_name)


if __name__ == "__main__":
    main(sys.argv)
