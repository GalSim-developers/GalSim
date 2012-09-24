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
- obj = galsim.RealGalaxy(real_galaxy_catalog, id)
- obj = galsim.Convolve([list], real_space)
- ps = galsim.PowerSpectrum(E_power_function, B_power_function)
- g1,g2 = ps.getShear(pos, grid_spacing, grid_nx, rng, center)

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
      - Applied shear is from a power spectrum P(k) ~ k^1.8.
      - Galaxies are randomly oriented real galaxies.
      - The PSF is Gaussian with fwhm,e,beta functions of (x,y)
      - Noise is poisson using a nominal sky value of 1.e6.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo10")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    n_tiles = 15                    # number of tiles in each direction.
    stamp_size = 48                 # pixels

    random_seed = 3339201           #

    pixel_scale = 0.44              # arcsec / pixel
    sky_level = 1.e6                # ADU / arcsec^2

    file_name = os.path.join('output','power_spectrum.fits')

    # These will be create for each object below.  The values we'll use will be functions
    # of (x,y) relative to the center of the image.  (r = sqrt(x^2+y^2))
    #psf_fwhm = 0.9 + 0.5 * (r/160)^2
    #psf_e = 0.4 * (r/160)^1.5
    #psf_beta = atan2(y/x) + pi/2

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

    # Setup the PowerSpectrum object we'll be using:
    ps = galsim.PowerSpectrum(lambda k : k**1.8)
    # The parameter here is E_power_function which defines the E-mode power to use.
    # There is also a B_power_function if you want to include any B-mode power:
    #ps = galsim.PowerSpectrum(E_power_function, B_power_function)
    # You may even omit the E_power_function argument and have a pure B-mode power spectrum.
    #ps = galsim.PowerSpectrum(B_power_function = B_power_function)

    # Now have it build a grid of shear values for us to use.
    # All the random number generator classes derive from BaseDeviate.
    # When we construct another kind of deviate class from any other
    # kind of deviate class, the two share the same underlying random number
    # generator.  Sometimes it can be clearer to just construct a BaseDeviate
    # explicitly and then construct anything else you need from that.
    # Note: A BaseDeviate cannot be used to generate any values.  It can
    # only be used in the constructor for other kinds of deviates.
    rng = galsim.BaseDeviate(random_seed)
    gal_g1, gal_g2 = ps.getShear(grid_spacing=stamp_size, grid_nx = n_tiles, rng=rng)
    print 'gal_g1 = ',gal_g1
    print 'gal_g2 = ',gal_g2

    # Setup the images:
    gal_image = galsim.ImageF(stamp_size * n_tiles , stamp_size * n_tiles)
    psf_image = galsim.ImageF(stamp_size * n_tiles , stamp_size * n_tiles)
    gal_image.setScale(pixel_scale)
    psf_image.setScale(pixel_scale)

    im_center = gal_image.bounds.center()
    #print 'image bounds = ',gal_image.bounds
    #print 'im_center = ',im_center

    # Build each postage stamp:
    k = 0
    for ix in range(n_tiles):
        for iy in range(n_tiles):
            # The seed here is augmented by k+1 rather than the usual k, since we already
            # used a seed for the power spectrum above.
            #rng = galsim.BaseDeviate(random_seed+k+1)
            rng = galsim.BaseDeviate(random_seed+k)
            print 'seed = ',random_seed+k

            # Determine the bounds for this stamp and its center position.
            b = galsim.BoundsI(ix*stamp_size+1 , (ix+1)*stamp_size, 
                               iy*stamp_size+1 , (iy+1)*stamp_size)
            sub_gal_image = gal_image[b]
            sub_psf_image = psf_image[b]

            pos = b.center() - im_center
            #print 'b.center = ',b.center()
            #print 'im_center = ',im_center
            #print 'pos = ',pos
            pos = galsim.PositionD(pos.x * pixel_scale , pos.y * pixel_scale)
            #print 'pos => ',pos
            # The image comes out as about 320 arcsec across, so we define our variable
            # parameters in terms of (r/160 arcsec), so roughly the scale size of the image.
            print 'pos = ',pos
            r = math.sqrt(pos.x**2 + pos.y**2) / 160
            print 'r = ',r
            psf_fwhm = 0.9 + 0.5 * r**2
            print 'fwhm = ',psf_fwhm
            psf_e = 0.4 * r**1.5
            print 'e = ',psf_e
            psf_beta = (math.atan2(pos.y,pos.x) + math.pi/2) * galsim.radians
            print 'beta = ',psf_beta

            # Define the PSF profile
            psf = galsim.Gaussian(fwhm=psf_fwhm)
            psf.applyShear(e=psf_e, beta=psf_beta)

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

            # Apply the shear from the power spectrum.
            print 'g1 = ',gal_g1[ix,iy]
            print 'g2 = ',gal_g2[ix,iy]
            #gal.applyShear(g1 = gal_g1[ix,iy], g2 = gal_g2[ix,iy])

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

            logger.info('Galaxy (%d,%d): center = (%.0f,%0.f)', ix,iy,pos.x,pos.y)
            k = k+1

    logger.info('Done making images of postage stamps')

    # Now write the images to disk.
    images = [ gal_image , psf_image ]
    galsim.fits.writeMulti(images, file_name)
    logger.info('Wrote image to %r',file_name) 

if __name__ == "__main__":
    main(sys.argv)
