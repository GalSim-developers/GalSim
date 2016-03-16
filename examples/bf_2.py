# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
bf_1.py

Modifying demo5.py to create 10 x 10 array of Gaussian spots
Craig Lage - 16-Mar-16

- Build a single large image, and access sub-images within it.
- Set the galaxy size based on the PSF size and a resolution factor.
- Set the object's flux according to a target S/N value.
- Shift by a random (dx, dy) drawn from a unit circle top hat.
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
    Make images similar to that done for the Great08 challenge:
      - Each fits file is 10 x 10 postage stamps.
        (The real Great08 images are 100x100, but in the interest of making the Demo
         script a bit quicker, we only build 100 stars and 100 galaxies.)
      - Each postage stamp is 40 x 40 pixels.
      - One image is all stars.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("bf_1")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    nx_tiles = 10                   #
    ny_tiles = 10                   #
    stamp_xsize = 40                #
    stamp_ysize = 40                #

    random_seed = 6424512           #

    pixel_scale = 0.2               # arcsec / pixel
    sky_level = 0.01                # ADU / arcsec^2

    # Make output directory if not already present.
    if not os.path.isdir('output'):
        os.mkdir('output')
    #psf_file_name = os.path.join('output','bf_1.psf')



    gal_sigma = 0.2     # arcsec
    psf_sigma = 0.01     # arcsec
    pixel_scale = 0.2  # arcsec / pixel
    noise = 0.01        # standard deviation of the counts in each pixel

    shift_radius = 0.2              # arcsec (=pixels)

    logger.info('Starting bf_1 using:')
    logger.info('    - image with %d x %d postage stamps',nx_tiles,ny_tiles)
    logger.info('    - postage stamps of size %d x %d pixels',stamp_xsize,stamp_ysize)
    logger.info('    - Centroid shifts up to = %.2f pixels',shift_radius)

    for nfile in range(1,6):

        gal_file_name = os.path.join('output','bf_%d.fits'%nfile)    
        gal_flux = 2.0e5 * nfile    # total counts on the image
        # Define the galaxy profile
        gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)
        logger.debug('Made galaxy profile')

        # Define the PSF profile
        psf = galsim.Gaussian(flux=1., sigma=psf_sigma) # PSF flux should always = 1
        logger.debug('Made PSF profile')

        # This profile is placed with different orientations and noise realizations
        # at each postage stamp in the gal image.
        gal_image = galsim.ImageF(stamp_xsize * nx_tiles-1 , stamp_ysize * ny_tiles-1,
                                  scale=pixel_scale)
        psf_image = galsim.ImageF(stamp_xsize * nx_tiles-1 , stamp_ysize * ny_tiles-1,
                                  scale=pixel_scale)

        shift_radius_sq = shift_radius**2

        first_in_pair = True  # Make pairs that are rotated by 90 degrees

        k = 0
        for iy in range(ny_tiles):
            for ix in range(nx_tiles):
                # The normal procedure for setting random numbers in GalSim is to start a new
                # random number generator for each object using sequential seed values.
                # This sounds weird at first (especially if you were indoctrinated by Numerical 
                # Recipes), but for the boost random number generator we use, the "random" 
                # number sequences produced from sequential initial seeds are highly uncorrelated.
                # 
                # The reason for this procedure is that when we use multiple processes to build
                # our images, we want to make sure that the results are deterministic regardless
                # of the way the objects get parcelled out to the different processes. 
                #
                # Of course, this script isn't using multiple processes, so it isn't required here.
                # However, we do it nonetheless in order to get the same results as the config
                # version of this demo script (demo5.yaml).
                ud = galsim.UniformDeviate(random_seed+k)

                # Any kind of random number generator can take another RNG as its first 
                # argument rather than a seed value.  This makes both objects use the same
                # underlying generator for their pseudo-random values.
                #gd = galsim.GaussianDeviate(ud, sigma=gal_ellip_rms)

                # The -1's in the next line are to provide a border of
                # 1 pixel between postage stamps
                b = galsim.BoundsI(ix*stamp_xsize+1 , (ix+1)*stamp_xsize-1, 
                                   iy*stamp_ysize+1 , (iy+1)*stamp_ysize-1)
                sub_gal_image = gal_image[b]
                sub_psf_image = psf_image[b]

                # Great08 randomized the locations of the two galaxies in each pair,
                # but for simplicity, we just do them in sequential postage stamps.

                if first_in_pair:
                    # Use a random orientation:
                    beta = ud() * 2. * math.pi * galsim.radians

                    # Determine the ellipticity to use for this galaxy.
                    ellip = 0.0
                    first_in_pair = False
                else:
                    # Use the previous ellip and beta + 90 degrees
                    beta += 90 * galsim.degrees
                    first_in_pair = True

                # Make a new copy of the galaxy with an applied e1/e2-type distortion 
                # by specifying the ellipticity and a real-space position angle
                this_gal = gal#gal.shear(e=ellip, beta=beta)

                # Apply a random shift_radius:
                rsq = 2 * shift_radius_sq
                while (rsq > shift_radius_sq):
                    dx = (2*ud()-1) * shift_radius
                    dy = (2*ud()-1) * shift_radius
                    rsq = dx**2 + dy**2

                this_gal = this_gal.shift(dx,dy)
                # Note that the shifted psf that we create here is purely for the purpose of being able
                # to draw a separate, shifted psf image.  We do not use it when convolving the galaxy
                # with the psf.
                this_psf = psf.shift(dx,dy)

                # Make the final image, convolving with the (unshifted) psf
                final_gal = galsim.Convolve([psf,this_gal])

                # Draw the image
                final_gal.drawImage(sub_gal_image, method = 'phot')

                # Now add an appropriate amount of noise to get our desired S/N
                # There are lots of definitions of S/N, but here is the one used by Great08
                # We use a weighted integral of the flux:
                #   S = sum W(x,y) I(x,y) / sum W(x,y)
                #   N^2 = Var(S) = sum W(x,y)^2 Var(I(x,y)) / (sum W(x,y))^2
                # Now we assume that Var(I(x,y)) is constant so
                #   Var(I(x,y)) = noise_var
                # We also assume that we are using a matched filter for W, so W(x,y) = I(x,y).
                # Then a few things cancel and we find that
                # S/N = sqrt( sum I(x,y)^2 / noise_var )
                #
                # The above procedure is encapsulated in the function image.addNoiseSNR which
                # sets the flux appropriately given the variance of the noise model.
                # In our case, noise_var = sky_level_pixel
                sky_level_pixel = sky_level * pixel_scale**2
                noise = galsim.PoissonNoise(ud, sky_level=sky_level_pixel)
                #sub_gal_image.addNoiseSNR(noise, gal_signal_to_noise)

                # Draw the PSF image
                # No noise on PSF images.  Just draw it as is.
                this_psf.drawImage(sub_psf_image)

                # For first instance, measure moments
                """
                if ix==0 and iy==0:
                    psf_shape = sub_psf_image.FindAdaptiveMom()
                    temp_e = psf_shape.observed_shape.e
                    if temp_e > 0.0:
                        g_to_e = psf_shape.observed_shape.g / temp_e
                    else:
                        g_to_e = 0.0
                    logger.info('Measured best-fit elliptical Gaussian for first PSF image: ')
                    logger.info('  g1, g2, sigma = %7.4f, %7.4f, %7.4f (pixels)',
                                g_to_e*psf_shape.observed_shape.e1,
                                g_to_e*psf_shape.observed_shape.e2, psf_shape.moments_sigma)
                """
                x = b.center().x
                y = b.center().y
                logger.info('Galaxy (%d,%d): center = (%.0f,%0.f)  (e,beta) = (%.4f,%.3f)',
                            ix,iy,x,y,ellip,beta/galsim.radians)
                k = k+1

        logger.info('Done making images of postage stamps')

        # Now write the images to disk.
        #psf_image.write(psf_file_name)
        #logger.info('Wrote PSF file %s',psf_file_name)

        gal_image.write(gal_file_name)
        logger.info('Wrote image to %r',gal_file_name)  # using %r adds quotes around filename for us


if __name__ == "__main__":
    main(sys.argv)
