"""
Demo #10

The tenth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script uses both a variable PSF and variable shear, taken from a power spectrum,
along the lines of a Great10 challenge image.  The galaxies are placed on a grid
(10 x 10 in this case, rather than 100 x 100 in the interest of time.)  Each postage stamp
is 48 x 48 pixels.  Instead of putting the psf images on a separate image, we package them
as the second hdu in the file.  For the galaxies, we use a random selection from 5 specific
RealGalaxy's, selected to be 5 particularly irregular ones, each with a random orientation.

New features introduced in this demo:

- rng = galsim.BaseDeviate(seed)

- Choosing PSF parameters as a function of (x,y)
- Selecting ReadGalaxy by ID rather than index.
- Putting the PSF image in a second hdu in the same file as the main image.
- Using PowerSpectrum for the applied shear.
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
    Make images using variable PSF and shear:
      - The main image is 10 x 10 postage stamps.
      - Each postage stamp is 48 x 48 pixels.
      - The second hdu has the corresponding PSF image.
      - Applied shear is from a power spectrum P(k) ~ k^2.
      - Galaxies are randomly oriented real galaxies.
      - The PSF is Moffat with e1,e2 polynomials in (x,y)
      - Noise is poisson using a nominal sky value of 1.e6.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo10")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    nx_tiles = 10                   #
    ny_tiles = 10                   #
    stamp_xsize = 48                #
    stamp_ysize = 48                #

    random_seed = 3339201           #

    pixel_scale = 0.44              # arcsec / pixel
    sky_level = 1.e6                # ADU / arcsec^2

    file_name = os.path.join('output','power_spectrum.fits')

    psf_beta = 2.4                  #
    psf_fwhm = 0.9                  # arcsec

    gal_signal_to_noise = 100       # Great08 "LowNoise" run
    gal_dilation = 3                # Make the galaxies a bit larger than their original size.

    logger.info('Starting demo script 10')
 
    # Read in galaxy catalog
    cat_file_name = 'real_galaxy_catalog_example.fits'
    image_dir = 'data'
    real_galaxy_catalog = galsim.RealGalaxyCatalog(cat_file_name, image_dir)
    real_galaxy_catalog.preload()
    logger.info('Read in %d real galaxies from catalog', real_galaxy_catalog.nobjects)

    # List of IDs to use.  We select 5 particularly irregular galaxies for this demo. 
    # Then we'll choose randomly from this list.
    id_list = [ 106416, 106731, 108402, 116045, 116448 ]

    # Make the 5 galaxies we're going to use here rather than remake them each time.
    # This means the Fourier transforms of the real galaxy images don't need to be recalculated 
    # each time, so it's a bit more efficient.
    gal_list = [ galsim.RealGalaxy(real_galaxy_catalog, id=id) for id in id_list ]

    # Make the galaxies a bit larger than their original oberved size.
    for gal in gal_list:
        gal.applyDilation(gal_dilation) 

    # Setup the images:
    gal_image = galsim.ImageF(stamp_xsize * nx_tiles , stamp_ysize * ny_tiles)
    psf_image = galsim.ImageF(stamp_xsize * nx_tiles , stamp_ysize * ny_tiles)
    gal_image.setScale(pixel_scale)
    psf_image.setScale(pixel_scale)

    # Build each postage stamp:
    k = 0
    for ix in range(nx_tiles):
        for iy in range(ny_tiles):
            # All the random number generator classes derive from BaseDeviate.
            # When we construct another kind of deviate class from any other
            # kind of deviate class, the two share the same underlying random number
            # generator.  Sometimes it can be clearer to just construct a BaseDeviate
            # explicitly and then construct anything else you need from that.
            # Note: A BaseDeviate cannot be used to generate any values.  It can
            # only be used in the constructor for other kinds of deviates.
            rng = galsim.BaseDeviate(random_seed+k)
            #print 'seed = ',random_seed+k

            # Determine the bounds for this stamp and its center position.
            b = galsim.BoundsI(ix*stamp_xsize+1 , (ix+1)*stamp_xsize, 
                               iy*stamp_ysize+1 , (iy+1)*stamp_ysize)
            sub_gal_image = gal_image[b]
            sub_psf_image = psf_image[b]
            x = b.center().x
            y = b.center().y
            #print 'x,y = ',x,y

            # Define the PSF profile
            psf = galsim.Moffat(psf_beta, fwhm = psf_fwhm)

            # Define the pixel
            pix = galsim.Pixel(pixel_scale)

            # Define the galaxy profile:
            ud = galsim.UniformDeviate(rng)
            index = int(ud() * len(gal_list))
            #print 'index = ',index
            gal = gal_list[index]

            # Apply a random rotation:
            theta = ud() * 360 * galsim.degrees 
            #print 'theta = ',theta
            # This makes a new copy so we're not changing the object in the gal_list.
            gal = gal.createRotated(theta)

            # Apply a random shift within a square box the size of a pixel
            dx = (ud()-0.5) * pixel_scale
            dy = (ud()-0.5) * pixel_scale
            #print 'shift  = ',galsim.PositionD(dx,dy)
            gal.applyShift(dx,dy)

            # Make the final image, convolving with psf and pixel
            final = galsim.Convolve([psf,pix,gal])

            # Draw the image
            final.draw(sub_gal_image)

            # Now determine what we need to do to get our desired S/N
            # See demo5.py for the math behind this calculation.
            sky_level_pix = sky_level * pixel_scale**2
            sn_meas = math.sqrt( numpy.sum(sub_gal_image.array**2) / sky_level_pix )
            flux = gal_signal_to_noise / sn_meas
            #print 'sn_meas = ',sn_meas
            #print 'flux = ',flux
            sub_gal_image *= flux

            # Add Poisson noise -- the CCDNoise can also take another rng as its argument
            # so it will be part of the same stream of random numbers as ud and gd.
            sub_gal_image += sky_level_pix
            sub_gal_image.addNoise(galsim.CCDNoise(rng))
            sub_gal_image -= sky_level_pix

            # Draw the PSF image
            # We use real space convolution to avoid some of the 
            # artifacts that can show up with Fourier convolution.
            # The level of the artifacts is quite low, but when drawing with
            # no noise, they are apparent with ds9's zscale viewing.
            final_psf = galsim.Convolve([psf,pix], real_space=True)

            # For the PSF image, we also shift the PSF by the same amount.
            final_psf.applyShift(dx,dy)

            # No noise on PSF images.  Just draw it as is.
            final_psf.draw(sub_psf_image)

            logger.info('Galaxy (%d,%d): center = (%.0f,%0.f)', ix,iy,x,y)
            k = k+1

    logger.info('Done making images of postage stamps')

    # Now write the images to disk.
    images = [ gal_image , psf_image ]
    galsim.fits.writeMulti(images, file_name)
    logger.info('Wrote image to %r',file_name) 

if __name__ == "__main__":
    main(sys.argv)
