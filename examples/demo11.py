"""
Demo #11

The eleventh script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script uses a constant PSF from real data (an image read in from a bzipped FITS file, not a
parametric model) and variable shear and magnification according to some cosmological model for
which we have a tabulated power spectrum at specific k values only.  The 288 galaxies in the 0.2 x
0.2 degree field (representing a low number density of 2/arcmin^2) are randomly located and
permitted to overlap, but we do take care to avoid being too close to the edge of the large image.
For the galaxies, we use a random selection from 5 specific RealGalaxy objects, selected to be 5
particularly irregular ones.  These are taken from the same catalog of 100 objects that demo6 used.

The noise added to the image is spatially correlated in the same way as often seen in coadd images
from the Hubble Space Telescope (HST) Advanced Camera for Surveys, using a correlation function
determined from the HST COSMOS coadd images in the F814W filter (see, e.g., Leauthaud et al 2007).
Applying this noise uses an FFT of the size of the full output image: this may cause memory-related
slowdowns in systems with less than 2GB RAM.

New features introduced in this demo:

- psf = galsim.InterpolatedImage(psf_filename, dx, flux)
- tab = galsim.LookupTable(file)
- ps = galsim.PowerSpectrum(..., units)
- distdev = galsim.DistDeviate(rng, function, x_min, x_max)
- gal.applyLensing(g1, g2, mu)
- cn = galsim.correlatednoise.getCOSMOSNoise(rng, file_name, ...)
- image.addNoise(cn)
- image.setOrigin(x,y)

- Power spectrum shears and magnifications for non-gridded positions.
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
      - The main image is 0.2 x 0.2 degrees.
      - Pixel scale is 0.2 arcsec, hence the image is 3600 x 3600 pixels.
      - Applied shear is from a cosmological power spectrum read in from file.
      - The PSF is a real one from SDSS, and corresponds to a convolution of atmospheric PSF,
        optical PSF, and pixel response, which has been sampled at pixel centers.  We used a PSF
        from SDSS in order to have a PSF profile that could correspond to what you see with a real
        telescope. However, in order that the galaxy resolution not be too poor, we tell GalSim that
        the pixel scale for that PSF image is 0.2" rather than 0.396".  We are simultaneously lying
        about the intrinsic size of the PSF and about the pixel scale when we do this.
      - Noise is correlated with the same spatial correlation function as found in HST COSMOS weak
        lensing science images, with a point (zero distance) variance that we normalize to 1.e4.
      - Galaxies are real galaxies, each with S/N~100 based on a point variance-only calculation
        (such as discussed in Leauthaud et al 2007).  The true SNR is somewhat lower, due to the
        presence of correlation in the noise.

    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo11")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    stamp_size = 100                  # number of pixels in each dimension of galaxy images
    pixel_scale = 0.2                 # arcsec/pixel
    image_size = 0.2 * galsim.degrees # size of big image in each dimension
    image_size = int((image_size / galsim.arcsec) / pixel_scale) # convert to pixels
    image_size_arcsec = image_size*pixel_scale # size of big image in each dimension (arcsec)
    noise_variance = 1.e4             # ADU^2
    nobj = 288                        # number of galaxies in entire field
                                      # (This corresponds to 2 galaxies / arcmin^2)
    grid_spacing = 90.0               # The spacing between the samples for the power spectrum 
                                      # realization (arcsec)
    gal_signal_to_noise = 100         # S/N of each galaxy

    # random_seed is used for both the power spectrum realization and the random properties
    # of the galaxies.
    random_seed = 24783923

    file_name = os.path.join('output','tabulated_power_spectrum.fits.fz')

    logger.info('Starting demo script 11')
 
    # Read in galaxy catalog
    cat_file_name = 'real_galaxy_catalog_example.fits'
    # This script is designed to be run from the examples directory so dir is a relative path.  
    # But the '../examples/' part lets bin/demo11 also be run from the bin directory.
    dir = '../examples/data'
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
    # terms of ell and C_ell; ell is inverse radians and C_ell in radians^2.  Since GalSim tends to
    # work in terms of arcsec, we have to tell it that the inputs are radians^-1 so it can convert
    # to store in terms of arcsec^-1.
    pk_file = os.path.join('..', 'examples', 'data','cosmo-fid.zmed1.00.out')
    ps = galsim.PowerSpectrum(pk_file, units = galsim.radians)
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
    psf_file = os.path.join('..', 'examples', 'data','example_sdss_psf_sky0.fits.bz2')
    psf = galsim.InterpolatedImage(psf_file, dx = pixel_scale, flux = 1.)
    # We do not include a pixel response function galsim.Pixel here, because the image that was read
    # in from file already included it.
    logger.info('Read in PSF image from bzipped FITS file')

    # Setup the image:
    full_image = galsim.ImageF(image_size, image_size)
    full_image.setScale(pixel_scale)

    # The default convention for indexing an image is to follow the FITS standard where the 
    # lower-left pixel is called (1,1).  However, this can be counter-intuitive to people more 
    # used to C or python indexing, where indices start at 0.  It is possible to change the 
    # coordinates of the lower-left pixel with the methods `setOrigin`.  For this demo, we
    # switch to 0-based indexing, so the lower-left pixel will be called (0,0).
    full_image.setOrigin(0,0)

    # Get the center of the image in arcsec
    center = full_image.bounds.trueCenter() * pixel_scale

    # As for demo10, we use random_seed+nobj for the random numbers required for the 
    # whole image.  In this case, both the power spectrum realization and the noise on the 
    # full image we apply later.
    rng = galsim.BaseDeviate(random_seed+nobj)
    # We want to make random positions within our image.  However, currently for shears from a power
    # spectrum we first have to get shears on a grid of positions, and then we can choose random
    # positions within that.  So, let's make the grid.  We're going to make it as large as the
    # image, with grid points spaced by 90 arcsec (hence interpolation only happens below 90"
    # scales, below the interesting scales on which we want the shear power spectrum to be
    # represented exactly).  Lensing engine wants positions in arcsec, so calculate that:
    ps.buildGrid(grid_spacing = grid_spacing,
                 ngrid = int(image_size_arcsec / grid_spacing)+1,
                 center = center, rng = rng)
    logger.info('Made gridded shears')

    # Now we need to loop over our objects:
    for k in range(nobj):
        time1 = time.time()
        # The usual random number generator using a different seed for each galaxy.
        ud = galsim.UniformDeviate(random_seed+k)

        # Choose a random position in the image
        x = ud()*(image_size-1)
        y = ud()*(image_size-1)

        # Turn this into a position in arcsec
        pos = galsim.PositionD(x,y) * pixel_scale
        
        # Get the reduced shears and magnification at this point
        g1, g2, mu = ps.getLensing(pos = pos)

        # Construct the galaxy:
        # Select randomly from among our list of galaxies.
        index = int(ud() * len(gal_list))
        gal = gal_list[index]

        # Draw the size from a plausible size distribution: N(r) ~ r^-3.5
        # For this, we use the class DistDeviate which can draw deviates from an arbitrary
        # probability distribution.  This distribution can be defined either as a functional
        # form as we do here, or as tabulated lists of x and p values, from which the 
        # function is interpolated.
        distdev = galsim.DistDeviate(ud, function=lambda x:x**-3.5, x_min=1, x_max=5)
        dilat = distdev()
        # Use createDilated rather than applyDilation, so we don't change the galaxies in the 
        # original gal_list -- createDilated makes a new copy.
        gal = gal.createDilated(dilat)

        # Apply a random rotation
        theta = ud()*2.0*numpy.pi*galsim.radians
        gal.applyRotation(theta)

        # Apply the cosmological (reduced) shear and magnification at this position using a single
        # GSObject method.
        gal.applyLensing(g1, g2, mu)

        # Convolve with the PSF.  We don't have to include a pixel response explicitly, since the
        # SDSS PSF image that we are using included the pixel response already.
        final = galsim.Convolve(psf, gal)

        # Account for the fractional part of the position:
        x_nom = x+0.5 # Because stamp size is even!  See discussion in demo9.py
        y_nom = y+0.5
        ix_nom = int(math.floor(x_nom+0.5))
        iy_nom = int(math.floor(y_nom+0.5))
        final.applyShift((x_nom-ix_nom)*pixel_scale,(y_nom-iy_nom)*pixel_scale)

        # Draw it with our desired stamp size
        stamp = galsim.ImageF(stamp_size,stamp_size)
        final.draw(image=stamp, dx=pixel_scale)

        # Rescale flux to get the S/N we want.  We have to do that before we add it to the big 
        # image, which might have another galaxy near that point (so our S/N calculation would 
        # erroneously include the flux from the other object).
        # See demo5.py for the math behind this calculation.
        sn_meas = math.sqrt( numpy.sum(stamp.array**2) / noise_variance )
        flux_scaling = gal_signal_to_noise / sn_meas
        stamp *= flux_scaling

        # Recenter the stamp at the desired position:
        stamp.setCenter(ix_nom,iy_nom)

        # Find the overlapping bounds:
        bounds = stamp.bounds & full_image.bounds
        full_image[bounds] += stamp[bounds]

        time2 = time.time()
        tot_time = time2-time1
        logger.info('Galaxy %d: position relative to corner = %s, t=%f s', k, str(pos), tot_time)

    # Add correlated noise to the image -- the correlation function comes from the HST COSMOS images
    # and is described in more detail in the galsim.correlatednoise.getCOSMOSNoise() docstring.
    # This function requires a FITS file, stored in the GalSim repository, that represents this
    # correlation information: the path to this file is a required argument. 
    cf_file_name = os.path.join('..', 'examples', 'data', 'acs_I_unrot_sci_20_cf.fits')

    # Then use this to initialize the correlation function that we will use to add noise to the
    # full_image.  We set the dx_cosmos keyword equal to our pixel scale, so that the noise among
    # neighboring pixels is correlated at the same level as it was among neighboring pixels in HST
    # COSMOS.  Using the original pixel scale, dx_cosmos=0.03 [arcsec], would leave very little
    # correlation among our larger 0.2 arcsec pixels. We also set the point (zero-distance) variance
    # to our desired value.
    cn = galsim.correlatednoise.getCOSMOSNoise(
        rng, cf_file_name, dx_cosmos=pixel_scale, variance=noise_variance)

    # Now add noise according to this correlation function to the full_image.  We have to do this
    # step at the end, rather than adding to individual postage stamps, in order to get the noise
    # level right in the overlap regions between postage stamps.
    full_image.addNoise(cn) # Note image must have the right scale [set by setScale()], as here
    logger.info('Added noise to final large image')

    # Now write the image to disk.  It is automatically compressed with Rice compression,
    # since the filename we provide ends in .fz.
    full_image.write(file_name)
    logger.info('Wrote image to %r',file_name) 

if __name__ == "__main__":
    main(sys.argv)
