"""
Demo #4

The fourth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script is our first one to create multiple images.  Typically, you would want each object
to have at least some of its attributes vary when you are drawing multiple images (although 
not necessarily -- you might just want different noise realization of the same profile).  
The easiest way to do this is to read in the properties from a catalog, which is what we
do in this script.  The PSF is a truncated Moffat profile, and the galaxy is bulge plus disk.
Both components get many of its parameters from an input catalog.  We also shift the 
profile by a fraction of a pixel in each direction so the effect of pixelization varies
among the images.  Each galaxy has the same applied shear.  The noise is simple Poisson noise.
We write the images out into a multi-extension fits file.

New features introduced in this demo:

- cat = galsim.InputCatalog(file_name, dir)
- obj = galsim.Moffat(beta, fwhm, trunc)
- obj = galsim.Add([list of objects])
- obj.setFlux(flux)
- galsim.fits.writeMulti([list of images], file_name)
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
    Make a fits image cube using parameters from an input catalog
      - The number of images in the cube matches the number of rows in the catalog.
      - Each image size is computed automatically by GalSim based on the Nyquist size.
      - Only galaxies.  No stars.
      - PSF is Moffat
      - Each galaxy is bulge plus disk: deVaucouleurs + Exponential.
      - The catalog's columns are:
         0 PSF beta
         1 PSF FWHM
         2 PSF e1
         3 PSF e2
         4 PSF trunc
         5 Disc half-light-radius
         6 Disc e1
         7 Disc e2
         8 Bulge half-light-radius
         9 Bulge e1
        10 Bulge e2
        11 Galaxy dx (the two components have same center)
        12 Galaxy dy
      - Applied shear is the same for each galaxy
      - Noise is poisson using a nominal sky value of 1.e6
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo4")

    # Define some parameters we'll use below.

    cat_file_name = os.path.join('input','galsim_default_input.asc')
    multi_file_name = os.path.join('output','multi.fits')

    random_seed = 8241573
    sky_level = 1.e6                # ADU / arcsec^2
    pixel_scale = 1.0               # arcsec / pixel  (size units in input catalog are pixels)
    gal_flux = 1.e6                 # arbitrary choise, makes nice (not too) noisy images
    gal_g1 = -0.009                 #
    gal_g2 = 0.011                  #
    xsize = 64                      # pixels
    ysize = 64                      # pixels

    logger.info('Starting demo script 4 using:')
    logger.info('    - parameters taken from catalog %r',cat_file_name)
    logger.info('    - Moffat PSF (parameters from catalog)')
    logger.info('    - pixel scale = %.2f',pixel_scale)
    logger.info('    - Bulge + Disc galaxies (parameters from catalog)')
    logger.info('    - Applied gravitational shear = (%.3f,%.3f)',gal_g1,gal_g2)
    logger.info('    - Poisson noise (sky level = %.1e).', sky_level)

    # Read in the input catalog
    cat = galsim.InputCatalog(cat_file_name)

    images = []
    for k in range(cat.nObjects()):

        # Take the Moffat beta from the first column (called 0) of the input catalog:
        # Note: cat.get(k,col) returns a string.  To get the value as a float, use either
        #       cat.getFloat(k,col) or float(cat.get(k,col))
        beta = cat.getFloat(k,0)
        # A Moffat's size may be either scale_radius, fwhm, or half_light_radius.
        # Here we use fwhm, taking from the catalog as well.
        fwhm = cat.getFloat(k,1)
        # A Moffat profile may be truncated if desired
        trunc = cat.getFloat(k,4)
        # Note: You may omit the flux, since the default is flux=1.
        psf = galsim.Moffat(beta=beta, fwhm=fwhm, trunc=trunc)

        # Take the (e1,e2) shape parameters from the catalog as well.
        psf.applyShear(e1=cat.getFloat(k,2), e2=cat.getFloat(k,3))

        pix = galsim.Pixel(pixel_scale)

        # Galaxy is a bulge + disk with parameters taken from the catalog:
        disk = galsim.Exponential(flux=0.6, half_light_radius=cat.getFloat(k,5))
        disk.applyShear(e1=cat.getFloat(k,6), e2=cat.getFloat(k,7))

        bulge = galsim.DeVaucouleurs(flux=0.4, half_light_radius=cat.getFloat(k,8))
        bulge.applyShear(e1=cat.getFloat(k,9), e2=cat.getFloat(k,10))

        # The flux of an Add object is the sum of the component fluxes
        gal = galsim.Add([disk,bulge])
        # This flux may be overridden by setFlux.  The relative fluxes of the components
        # remains the same, but the total flux is set to gal_flux.
        gal.setFlux(gal_flux)
        gal.applyShear(g1=gal_g1, g2=gal_g2)

        # The center of the object is normally placed at the center of the postage stamp image.
        # You can change that with applyShift:
        gal.applyShift(dx=cat.getFloat(k,11), dy=cat.getFloat(k,12))

        final = galsim.Convolve([psf, pix, gal])

        # Draw the profile
        image = final.draw(image = galsim.ImageF(xsize, ysize), dx=pixel_scale)

        # Add poisson noise to the image:
        image += sky_level * pixel_scale**2
        # The default gain, read_noise are 1 and 0, so we can omit them.
        # Also, use a different random seed for each object to get different noise realizations.
        image.addNoise(galsim.CCDNoise(random_seed+k))
        image -= sky_level * pixel_scale**2

        logger.info('Drew image for object at row %d in the input catalog'%k)
   
        # Add the image to our list of images
        images.append(image)
    
    # Now write the images to a multi-extension fits file.  Each image will be in its own HDU.
    galsim.fits.writeMulti(images, multi_file_name)
    logger.info('Images written to multi-extension fits file %r',multi_file_name)


if __name__ == "__main__":
    main(sys.argv)
