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
tr_plot.py

Modifying demo5.py to create arrays of Gaussian spots
for characterizing the tree rings

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
    Make images to be used for characterizing the brighter-fatter effect
      - Each fits file is 10 x 10 postage stamps.
      - Each postage stamp is 40 x 40 pixels.
      - There are 2 fits files, one with tree rings and one without
      - Each image is in output/tr_nfile.fits, where nfile ranges from 1-2.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("tr_plots")

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

    gal_sigma = 1.0     # pixels
    psf_sigma = 0.01     # pixels
    pixel_scale = 1.0
    noise = 0.01        # standard deviation of the counts in each pixel

    logger.info('Starting tr_plots using:')
    logger.info('    - image with %d x %d postage stamps',nx_tiles,ny_tiles)
    logger.info('    - postage stamps of size %d x %d pixels',stamp_xsize,stamp_ysize)

    rng = galsim.BaseDeviate(5678)    
    sensor1 = galsim.SiliconSensor(rng=rng, diffusion_factor=0.0)
    # The following makes a sensor with hugely magnified tree rings so you can see the effect
    sensor2 = galsim.SiliconSensor(rng=rng, diffusion_factor = 0.0, treeringamplitude = 1.00)

    for nfile in range(1,3):
        starttime = time.time()
        exec("sensor = sensor%d"%nfile)

        gal_file_name = os.path.join('output','tr_%d.fits'%nfile)
        sex_file_name = os.path.join('output','tr_%d_SEX.fits.cat.reg'%nfile)
        sexfile = open(sex_file_name, 'w')
        gal_flux = 2.0e5    # total counts on the image
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
                #print "ix = %d, iy = %d, cenx = %d, ceny = %d"%(ix,iy,sub_gal_image.center().x,sub_gal_image.center().y)
                # Make the final image, convolving with the (unshifted) psf
                final_gal = galsim.Convolve([psf,gal])

                # Draw the image
                final_gal.drawImage(sub_gal_image, method = 'phot', sensor=sensor, rng = rng)

                x = b.center().x
                y = b.center().y
                k = k+1
                sexline = 'circle %f %f %f\n'%(x,y,gal_sigma/pixel_scale)
                sexfile.write(sexline)

        sexfile.close()
        logger.info('Done making images of postage stamps')

        # Now write the images to disk.
        #psf_image.write(psf_file_name)
        #logger.info('Wrote PSF file %s',psf_file_name)

        gal_image.write(gal_file_name)
        logger.info('Wrote image to %r',gal_file_name)  # using %r adds quotes around filename for us

        finishtime = time.time()
        print("Time to complete file %d = %.2f seconds\n"%(nfile, finishtime-starttime))
if __name__ == "__main__":
    main(sys.argv)
