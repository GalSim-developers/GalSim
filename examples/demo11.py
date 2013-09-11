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

New features introduced in this demo:

- psf = galsim.InterpolatedImage(psf_filename, dx, flux)
- tab = galsim.LookupTable(file)
- gal = galsim.RealGalaxy(..., noise_pad_size)
- ps = galsim.PowerSpectrum(..., units)
- distdev = galsim.DistDeviate(rng, function, x_min, x_max)
- gal.applyLensing(g1, g2, mu)
- correlated_noise.applyWhiteningTo(image)
- vn = galsim.VariableGaussianNoise(rng, var_image)
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
      - The galaxy images include some initial correlated noise from the original HST observation.
        However, we whiten the noise of the final image so the final image has stationary 
        Gaussian noise, rather than correlated noise.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo11")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    base_stamp_size = 32              # number of pixels in each dimension of galaxy images
                                      # This will be scaled up according to the dilation.
                                      # Hence the "base_" prefix.

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
    dir = 'data'
    real_galaxy_catalog = galsim.RealGalaxyCatalog(cat_file_name, dir=dir)
    real_galaxy_catalog.preload()
    logger.info('Read in %d real galaxies from catalog', real_galaxy_catalog.nobjects)

    # List of IDs to use.  We select 5 particularly irregular galaxies for this demo. 
    # Then we'll choose randomly from this list.
    id_list = [ 106416, 106731, 108402, 116045, 116448 ]

    # We will cache the galaxies that we make in order to save some of the calculations that
    # happen on construction.  In particular, we don't want to recalculate the Fourier transforms 
    # of the real galaxy images, so it's more efficient so make a store of RealGalaxy instances.
    # We start with them all = None, and fill them in as we make them.
    gal_list = [ None ] * len(id_list)

    # Setup the PowerSpectrum object we'll be using:
    # To do this, we first have to read in the tabulated shear power spectrum, often denoted
    # C_ell(ell), where ell has units of inverse angle and C_ell has units of angle^2.  However,
    # GalSim works in the flat-sky approximation, so we use this notation interchangeably with
    # P(k).  GalSim does not calculate shear power spectra for users, who must be able to provide
    # their own (or use the examples in the repository).
    # 
    # Here we use a tabulated power spectrum from iCosmo (http://icosmo.org), with the following
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
    pk_file = os.path.join('data','cosmo-fid.zmed1.00.out')
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
    psf_file = os.path.join('data','example_sdss_psf_sky0.fits.bz2')
    psf = galsim.InterpolatedImage(psf_file, dx = pixel_scale, flux = 1.)
    # We do not include a pixel response function galsim.Pixel here, because the image that was read
    # in from file already included it.
    logger.info('Read in PSF image from bzipped FITS file')

    # Setup the image:
    full_image = galsim.ImageF(image_size, image_size, scale=pixel_scale)

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

    # We keep track of how much noise is already in the image from the RealGalaxies.
    # The default initial value is all pixels = 0.
    noise_image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
    noise_image.setOrigin(0,0)

    # Now we need to loop over our objects:
    for k in range(nobj):
        time1 = time.time()
        # The usual random number generator using a different seed for each galaxy.
        ud = galsim.UniformDeviate(random_seed+k)

        # Draw the size from a plausible size distribution: N(r) ~ r^-2.5
        # For this, we use the class DistDeviate which can draw deviates from an arbitrary
        # probability distribution.  This distribution can be defined either as a functional
        # form as we do here, or as tabulated lists of x and p values, from which the 
        # function is interpolated.
        # N.B. This calculation logically belongs later in the script, but given how the config 
        #      structure works and the fact that we also use this value for the stamp size 
        #      calculation, in order to get the output file to match the YAML output file, it
        #      turns out this is where we need to put this use of the random number generator.
        distdev = galsim.DistDeviate(ud, function=lambda x:x**-2.5, x_min=1, x_max=5)
        dilat = distdev()

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

        # If we haven't made this galaxy yet, we need to do so.
        if gal is None:
            # When whitening the image, we need to make sure the original correlated noise is
            # present throughout the whole image, otherwise the whitening will do the wrong thing
            # to the parts of the image that don't include the original image.  The RealGalaxy
            # stores the correct noise profile to use as the gal.noise attribute.  This noise
            # profile is automatically updated as we shear, dilate, convolve, etc.  But we need to 
            # tell it how large to pad with this noise by hand.  This is a bit complicated for the 
            # code to figure out on its own, so we have to supply the size for noise padding 
            # with the noise_pad_size parameter.
        
            # In this case, the postage stamp will be 32 pixels for the undilated galaxies. 
            # We expand the postage stamp as we dilate the galaxies, so that factor doesn't
            # come into play here.  The shear and magnification are not significant, but the 
            # image can be rotated, which adds an extra factor of sqrt(2). So the net required 
            # padded size is
            #     noise_pad_size = 32 * sqrt(2) * 0.2 arcsec/pixel = 9.1 arcsec
            # We round this up to 10 to be safe.
            gal = galsim.RealGalaxy(real_galaxy_catalog, rng=ud, id=id_list[index],
                                    noise_pad_size=10) 
            # Save it for next time we use this galaxy.
            gal_list[index] = gal

        # Apply the dilation we calculated above.
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
        ix = int(math.floor(x+0.5))
        iy = int(math.floor(y+0.5))
        offset = galsim.PositionD(x-ix, y-iy)

        # Draw it with our desired stamp size (scaled up by the dilation factor):
        # Note: We make the stamp size odd to make the above calculation of the offset easier.
        this_stamp_size = 2 * int(math.ceil(base_stamp_size * dilat / 2)) + 1
        stamp = galsim.ImageF(this_stamp_size,this_stamp_size)
        final.draw(image=stamp, dx=pixel_scale, offset=offset)

        # Now we can whiten the noise on the postage stamp.
        # Galsim automatically propagates the noise correctly from the initial RealGalaxy object
        # through the applied shear, distortion, rotation, and convolution into the final object's
        # noise attribute.
        # The returned value is the variance of the Gaussian noise that is present after
        # the whitening process.
        new_variance = final.noise.applyWhiteningTo(stamp)

        # Rescale flux to get the S/N we want.  We have to do that before we add it to the big 
        # image, which might have another galaxy near that point (so our S/N calculation would 
        # erroneously include the flux from the other object).
        # See demo5.py for the math behind this calculation.
        sn_meas = math.sqrt( numpy.sum(stamp.array**2) / noise_variance )
        flux_scaling = gal_signal_to_noise / sn_meas
        stamp *= flux_scaling
        # This also scales up the current variance by flux_scaling**2.
        new_variance *= flux_scaling**2

        # Recenter the stamp at the desired position:
        stamp.setCenter(ix,iy)

        # Find the overlapping bounds:
        bounds = stamp.bounds & full_image.bounds
        full_image[bounds] += stamp[bounds]

        # We need to keep track of how much variance we have currently in the image, so when
        # we add more noise, we can omit what is already there.
        noise_image[bounds] += new_variance

        time2 = time.time()
        tot_time = time2-time1
        logger.info('Galaxy %d: position relative to corner = %s, t=%f s', k, str(pos), tot_time)

    # We already have some noise in the image, but it isn't uniform.  So the first thing to do is
    # to make the Gaussian noise uniform across the whole image.  We have a special noise class
    # that can do this.  VariableGaussianNoise takes an image of variance values and applies
    # Gaussian noise with the corresponding variance to each pixel.
    # So all we need to do is build an image with how much noise to add to each pixel to get us
    # up to the maximum value that we already have in the image.
    max_current_variance = numpy.max(noise_image.array)
    noise_image = max_current_variance - noise_image
    vn = galsim.VariableGaussianNoise(rng, noise_image)
    full_image.addNoise(vn)

    # Now max_current_variance is the noise level across the full image.  We don't want to add that
    # twice, so subtract off this much from the intended noise that we want to end up in the image.
    noise_variance -= max_current_variance

    # Now add Gaussian noise with this variance to the final image.  We have to do this step
    # at the end, rather than adding to individual postage stamps, in order to get the noise
    # level right in the overlap regions between postage stamps.
    noise = galsim.GaussianNoise(rng, sigma=math.sqrt(noise_variance))
    full_image.addNoise(noise)
    logger.info('Added noise to final large image')

    # Now write the image to disk.  It is automatically compressed with Rice compression,
    # since the filename we provide ends in .fz.
    full_image.write(file_name)
    logger.info('Wrote image to %r',file_name) 

if __name__ == "__main__":
    main(sys.argv)
