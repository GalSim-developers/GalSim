"""
Demo #11

The eleventh script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script uses a constant PSF from real data (an image read in from a bzipped FITS file, not a
parametric model) and variable shear according to some cosmological model for which we have a
tabulated power spectrum at specific k values only.  The 100 galaxies in the 0.25x0.25 degree field
(representing a low number density of 0.44/arcmin^2) are randomly located and permitted to overlap, but
we do take care to avoid being too close to the edge of the large image.  For the galaxies, we use a
random selection from 5 specific RealGalaxy objects, selected to be 5 particularly irregular
ones. These are taken from the same catalog of 100 objects that demo6 used.

The process of drawing the galaxies in this demo is particularly slow because it is the most
difficult possible case: realistic galaxy images and a non-parametric PSF profile.  However, because
we have a relatively large image in the end, and we want to compress the file, the process of
writing the results to file is also slow.

New features introduced in this demo:

- psf = galsim.InterpolatedImage(psf_filename, ...)
- tab = galsim.lensing.TabulatedPk(file, ...)
- Power spectrum shears for non-gridded positions.
- Reading a compressed FITS image (using BZip2 compression).
- Writing a compressed FITS image (using Rice compression).
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
    Make images using constant PSF and variable shear:
      - The main image is 0.25 x 0.25 degrees.
      - Pixel scale is 0.2 arcsec, hence the image is 4500 x 4500 pixels.
      - Applied shear is from a cosmological power spectrum read in from file.
      - The PSF is a real one from SDSS but, in order that the galaxy resolution not be too poor, we
        tell GalSim that the pixel scale for that PSF image is 0.2" rather than 0.396".
      - Galaxies are real galaxies, each with S/N~100.
      - Noise is Poisson using a nominal sky value of 1.e4.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo11")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    image_size = 0.25*galsim.degrees # size of big image in each dimension
    stamp_size = 48                  # number of pixels in each dimension of galaxy images
    pixel_scale = 0.2                # arcsec/pixel
    total_image_size = int((image_size / galsim.arcsec)/pixel_scale) # size of big image in each
                                                                     # dimension (pixels)
    total_image_size_arcsec = total_image_size*pixel_scale # size of big image in each dimension
                                                           # (arcsec)
    sky_level = 1.e4                 # ADU / arcsec^2
    nobj = 100                       # number of galaxies in entire field

    # random_seed is used for the power spectrum realization.
    # random_seed_gal is used for the galaxy properties.
    random_seed = 3339201
    random_seed_gal = 1234567

    file_name = os.path.join('output','tabulated_power_spectrum.fits.fz')

    gal_signal_to_noise = 100

    logger.info('Starting demo script 11')
 
    # Read in galaxy catalog
    cat_file_name = 'real_galaxy_catalog_example.fits'
    dir = 'data'
    real_galaxy_catalog = galsim.RealGalaxyCatalog(cat_file_name, dir=dir)
    real_galaxy_catalog.preload()
    logger.info('Read in %d real galaxies from catalog', real_galaxy_catalog.nobjects)

    # List of IDs to use.  We select 5 particularly irregular galaxies for this demo. 
    # Then we'll choose randomly from this list.
    id_list = [ 106416, 106731, 108402, 116045, 116448 ]

    # Make the 5 galaxies we're going to use here rather than remake them each time.
    # This means the Fourier transforms of the real galaxy images don't need to be recalculated 
    # each time, so it's a bit more efficient.
    gal_list = [ galsim.RealGalaxy(real_galaxy_catalog, id=id) for id in id_list ]

    # Setup the PowerSpectrum object we'll be using:
    # To do this, we first have to read in the tabulated power spectrum.
    # We use a tabulated power spectrum from iCosmo (http://icosmo.org), with the following
    # cosmological parameters and survey design:
    # H_0 = 70 km/s/Mpc
    # Omega_m = 0.25
    # Omega_Lambda = 0.75
    # w_0 = -1.0
    # w_a = 0.0
    # n_s = 0.96
    # sigma_8 = 0.8
    # Smith et al. prescription for the non-linear power spectrum.
    # Eisenstein & Hu transfer function with wiggles.
    # Default dN/dz with z_med = 1.0
    # The file has, as required, just two columns which are k and P(k).  However, iCosmo works in
    # terms of ell and C_ell; ell is inverse radians.  Since GalSim tends to work in terms of
    # arcsec, we have to tell it that the inputs are radians^-1 so it can convert to store in terms
    # of arcsec^-1.  Also, we need to tell GalSim that it is getting the C_ell (i.e., Delta^2) so it
    # can convert to power.
    pk_file = os.path.join('data','cosmo-fid.zmed1.00.out')
    tab_pk = galsim.lensing.TabulatedPk(file = pk_file, delta2 = True, units = galsim.radians)
    ps = galsim.PowerSpectrum(tab_pk)
    # The argument here is "e_power_function" which defines the E-mode power to use.
    logger.info('Set up power spectrum from tabulated P(k)')

    # Now let's read in the PSF.  It's a real SDSS PSF, which means pixel scale of 0.396".  However,
    # the typical seeing is 1.2" and we want to simulate better seeing, so we will just tell GalSim
    # that the pixel scale is 0.2".  We have to be careful with SDSS PSF images, as they have an
    # added 'soft bias' of 1000 which has been removed before creation of this file, so that the sky
    # level is properly zero.  Also, the file is bzipped, to demonstrate the new capability of
    # reading in a file that has been compressed in various ways (which GalSim can infer from the
    # filename).  We want to read the image directly into an InterpolatedImage GSObject, so we can
    # manipulate it as needed (here, the only manipulation needed is convolution).  We want a PSF
    # with flux 1, and we can set the pixel scale using a keyword.
    psf_file = os.path.join('data','example_sdss_psf_sky0.fits.bz2')
    psf = galsim.InterpolatedImage(psf_file, dx = pixel_scale, flux = 1.)
    logger.info('Read in PSF image from bzipped FITS file')

    # We want to make random positions within our image.  However, currently for shears from a power
    # spectrum we first have to get shears on a grid of positions, and then we can choose random
    # positions within that.  So, let's make the grid.  We're going to make it as large as the
    # image, with grid points spaced by 10 arcsec (hence interpolation only happens below 10"
    # scales, below the interesting scales on which we want the shear power spectrum to be
    # represented exactly).  Lensing engine wants positions in arcsec, so calculate that:
    rng = galsim.BaseDeviate(random_seed)
    g1_grid, g2_grid = ps.getShear(grid_spacing=10.0, ngrid=total_image_size_arcsec, rng=rng)
    logger.info('Made gridded shears')

    # Setup the image:
    gal_image = galsim.ImageF(total_image_size, total_image_size)
    gal_image.setScale(pixel_scale)

    # Now we need to loop over our objects:
    rng = galsim.BaseDeviate(random_seed_gal)
    ud = galsim.UniformDeviate(rng)
    for k in range(nobj):
        time1 = time.time()
        # Choose a random position within a range that is not too close to the edge.
        # Define this in arcsec and in pixel, rounding to the nearest pixel.  We are not including
        # sub-pixel shifts in this demo.
        x_pos = int(0.5*stamp_size + ud()*(total_image_size - stamp_size))
        y_pos = int(0.5*stamp_size + ud()*(total_image_size - stamp_size))
        x_pos_arcsec = x_pos*pixel_scale
        y_pos_arcsec = y_pos*pixel_scale

        # Loop over the galaxies within our list of 5.
        index = k % 5
        gal = gal_list[index]

        # Get the shear at this position.
        g1, g2 = ps.getShear(pos = (x_pos_arcsec, y_pos_arcsec))

        # Construct the galaxy:
        #   Random rotation
        theta = ud()*2.0*numpy.pi*galsim.radians
        gal = gal.createRotated(theta)
        #   Apply the cosmological shear
        gal.applyShear(g1 = g1, g2 = g2)
        #   Convolve with the PSF.  We don't have to include a pixel response explicitly, since the
        #     SDSS PSF image that we are using included the pixel response already.
        final = galsim.Convolve(gal, psf)

        # Identify the bounds within which to draw the image.
        b = galsim.BoundsI(int(x_pos - 0.5*stamp_size), int(x_pos + 0.5*stamp_size - 1),
                           int(y_pos - 0.5*stamp_size), int(y_pos + 0.5*stamp_size - 1))
        sub_gal_image = gal_image[b]
        tmp_gal_image = galsim.ImageF(b)
        tmp_gal_image.setScale(pixel_scale)

        # Draw it; we first write into its own postage stamp so we can properly rescale flux to get
        # the S/N we want.  We have to do that before we add it to the big image, which might have
        # another galaxy near that point (so our S/N calculation would erroneously include the flux
        # from the other object).
        final.draw(tmp_gal_image)

        # Now determine what we need to do to get our desired S/N
        # See demo5.py for the math behind this calculation.
        sky_level_pix = sky_level * pixel_scale**2
        sn_meas = math.sqrt( numpy.sum(tmp_gal_image.array**2) / sky_level_pix )
        flux_scaling = gal_signal_to_noise / sn_meas
        tmp_gal_image *= flux_scaling

        # Now that we have scaled the galaxy flux properly, add it to the big image in the proper
        # location.
        sub_gal_image += tmp_gal_image
        time2 = time.time()
        tot_time = time2-time1
        logger.info('Galaxy %d: position relative to corner = %s, t=%f s', k,
                    str(galsim.PositionI(x_pos, y_pos)), tot_time)

    # Add Poisson noise -- the CCDNoise can also take another RNG as its argument
    # so it will be part of the same stream of random numbers as ud above.  We have to do this step
    # at the end, rather than adding to individual postage stamps, in order to get the noise level
    # right in the overlap regions between postage stamps.
    gal_image += sky_level_pix
    gal_image.addNoise(galsim.CCDNoise(rng))
    gal_image -= sky_level_pix
    logger.info('Added noise to final large image')

    # Now write the image to disk.  It's automatically compressed with Rice compression,
    # since the filename we provide ends in .fz
    galsim.fits.write(gal_image, file_name)
    logger.info('Wrote image to %r',file_name) 

if __name__ == "__main__":
    main(sys.argv)
