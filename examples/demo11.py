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
""""
Demo #11

The eleventh script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script uses a constant PSF from real data (an image read in from a bzipped FITS file, not a
parametric model) and variable shear and magnification according to some cosmological model for
which we have a tabulated shear power spectrum at specific k values only.  The 288 galaxies in the
0.11 x 0.11 degree field (representing a number density of 6/arcmin^2) are randomly located and
permitted to overlap.  For the galaxies, we use a mix of real and parametric galaxies modeled off
the COSMOS observations with the Hubble Space Telescope.  The real galaxies are similar to those
used in demo10.  The parametric galaxies are based on parametric fits to the same observed galaxies.
The flux and size distribution are thus realistic for an I < 23.5 magnitude limited sample.

New features introduced in this demo:

- coord = galsim.CelestialCoord(ra, dec)
- wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin)
- wcs = galsim.TanWCS(affine, world_origin, units)
- psf = galsim.InterpolatedImage(psf_filename, scale, flux)
- tab = galsim.LookupTable(file)
- cosmos_cat = galsim.COSMOSCatalog(file_name, dir)
- gal = cosmos_cat.makeGalaxy(gal_type, rng, noise_pad_size)
- ps = galsim.PowerSpectrum(..., units)
- gal = gal.lens(g1, g2, mu)
- image.whitenNoise(correlated_noise)
- image.symmetrizeNoise(correlated_noise)
- vn = galsim.VariableGaussianNoise(rng, var_image)
- image.addNoise(cn)
- image.setOrigin(x,y)
- angle.dms(), angle.hms()

- Power spectrum shears and magnifications for non-gridded positions.
- Reading a compressed FITS image (using BZip2 compression).
- Writing a compressed FITS image (using Rice compression).
- Writing WCS information to a FITS header that ds9 reads as RA, Dec
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
      - The main image is 2048 x 2048 pixels.
      - Pixel scale is 0.2 arcsec/pixel, hence the image is about 0.11 degrees on a side.
      - Applied shear is from a cosmological power spectrum read in from file.
      - The PSF is a real one from SDSS, and corresponds to a convolution of atmospheric PSF,
        optical PSF, and pixel response, which has been sampled at pixel centers.  We used a PSF
        from SDSS in order to have a PSF profile that could correspond to what you see with a real
        telescope. However, in order that the galaxy resolution not be too poor, we tell GalSim that
        the pixel scale for that PSF image is 0.2" rather than 0.396".  We are simultaneously lying
        about the intrinsic size of the PSF and about the pixel scale when we do this.
      - The galaxies come from COSMOSCatalog, which can produce either RealGalaxy profiles
        (like in demo10) and parametric fits to those profiles.  We choose 30% of the galaxies
        to use the images, and the other 60% to use the parametric fits
      - The real galaxy images include some initial correlated noise from the original HST
        observation.  However, we whiten the noise of the final image so the final image has
        stationary Gaussian noise, rather than correlated noise.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo11")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    pixel_scale = 0.2                 # arcsec/pixel
    image_size = 2048                 # size of image in pixels
    image_size_arcsec = image_size*pixel_scale # size of big image in each dimension (arcsec)
    noise_variance = 5.e4             # ADU^2  (Just use simple Gaussian noise here.)
    nobj = 288                        # number of galaxies in entire field
                                      # (This corresponds to 8 galaxies / arcmin^2)
    grid_spacing = 90.0               # The spacing between the samples for the power spectrum
                                      # realization (arcsec)
    tel_diam = 4                      # Let's figure out the flux for a 4 m class telescope
    exp_time = 300                    # exposing for 300 seconds.
    center_ra = 19.3*galsim.hours     # The RA, Dec of the center of the image on the sky
    center_dec = -33.1*galsim.degrees

    # The catalog returns objects that are appropriate for HST in 1 second exposures.  So for our
    # telescope we scale up by the relative area and exposure time.  Note that what is important is
    # the *effective* area after taking into account obscuration.  For HST, the telescope diameter
    # is 2.4 but there is obscuration (a linear factor of 0.33).  Here, we assume that the telescope
    # we're simulating effectively has no obscuration factor.  We're also ignoring the pi/4 factor
    # since it appears in the numerator and denominator, so we use area = diam^2.
    hst_eff_area = 2.4**2 * (1.-0.33**2)
    flux_scaling = (tel_diam**2/hst_eff_area) * exp_time

    # random_seed is used for both the power spectrum realization and the random properties
    # of the galaxies.
    random_seed = 24783923

    file_name = os.path.join('output','tabulated_power_spectrum.fits.fz')

    logger.info('Starting demo script 11')

    # Read in galaxy catalog
    # The COSMOSCatalog uses the same input file as we have been using for RealGalaxyCatalogs
    # along with a second file called real_galaxy_catalog_23.5_examples_fits.fits, which stores
    # the information about the parameteric fits.  There is no need to specify the second file
    # name, since the name is derivable from the name of the main catalog.
    if True:
        # The catalog we distribute with the GalSim code only has 100 galaxies.
        # The galaxies will typically be reused several times here.
        cat_file_name = 'real_galaxy_catalog_23.5_example.fits'
        dir = 'data'
        cosmos_cat = galsim.COSMOSCatalog(cat_file_name, dir=dir)
    else:
        # If you've run galsim_download_cosmos, you can leave out the cat_file_name and dir
        # to use the full COSMOS catalog with 56,000 galaxies in it.
        cosmos_cat = galsim.COSMOSCatalog()
    logger.info('Read in %d galaxies from catalog', cosmos_cat.nobjects)

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
    # level is properly zero.  Also, the file is bzipped, to demonstrate the ability of GalSim
    # handle this kind of compressed file (among others).  We read the image directly into an
    # InterpolatedImage GSObject, so we can manipulate it as needed (here, the only manipulation
    # needed is convolution).  The flux is 1 as needed for a PSF.
    psf_file = os.path.join('data','example_sdss_psf_sky0.fits.bz2')
    psf = galsim.InterpolatedImage(psf_file, scale = pixel_scale, flux = 1.)
    logger.info('Read in PSF image from bzipped FITS file')

    # Setup the image:
    full_image = galsim.ImageF(image_size, image_size)

    # The default convention for indexing an image is to follow the FITS standard where the
    # lower-left pixel is called (1,1).  However, this can be counter-intuitive to people more
    # used to C or python indexing, where indices start at 0.  It is possible to change the
    # coordinates of the lower-left pixel with the methods `setOrigin`.  For this demo, we
    # switch to 0-based indexing, so the lower-left pixel will be called (0,0).
    full_image.setOrigin(0,0)

    # As for demo10, we use random_seed for the random numbers required for the
    # whole image.  In this case, both the power spectrum realization and the noise on the
    # full image we apply later.
    rng = galsim.BaseDeviate(random_seed)

    # We want to make random positions within our image.  However, currently for shears from a power
    # spectrum we first have to get shears on a grid of positions, and then we can choose random
    # positions within that.  So, let's make the grid.  We're going to make it as large as the
    # image, with grid points spaced by 90 arcsec (hence interpolation only happens below 90"
    # scales, below the interesting scales on which we want the shear power spectrum to be
    # represented exactly).  The lensing engine wants positions in arcsec, so calculate that:
    ps.buildGrid(grid_spacing = grid_spacing,
                 ngrid = int(math.ceil(image_size_arcsec / grid_spacing))+1, rng=rng)
    logger.info('Made gridded shears')

    # We keep track of how much noise is already in the image from the RealGalaxies.
    # The default initial value is all pixels = 0.
    noise_image = galsim.ImageF(image_size, image_size)
    noise_image.setOrigin(0,0)

    # Make a slightly non-trivial WCS.  We'll use a slightly rotated coordinate system
    # and center it at the image center.
    theta = 0.17 * galsim.degrees
    # ( dudx  dudy ) = ( cos(theta)  -sin(theta) ) * pixel_scale
    # ( dvdx  dvdy )   ( sin(theta)   cos(theta) )
    # Aside: You can call numpy trig functions on Angle objects directly, rather than getting
    #        their values in radians first.  Or, if you prefer, you can write things like
    #        theta.sin() or theta.cos(), which are equivalent.
    dudx = numpy.cos(theta) * pixel_scale
    dudy = -numpy.sin(theta) * pixel_scale
    dvdx = numpy.sin(theta) * pixel_scale
    dvdy = numpy.cos(theta) * pixel_scale
    image_center = full_image.true_center
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)

    # We can also put it on the celestial sphere to give it a bit more realism.
    # The TAN projection takes a (u,v) coordinate system on a tangent plane and projects
    # that plane onto the sky using a given point as the tangent point.  The tangent
    # point should be given as a CelestialCoord.
    sky_center = galsim.CelestialCoord(ra=center_ra, dec=center_dec)

    # The third parameter, units, defaults to arcsec, but we make it explicit here.
    # It sets the angular units of the (u,v) intermediate coordinate system.
    wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
    full_image.wcs = wcs

    # Now we need to loop over our objects:
    for k in range(nobj):
        time1 = time.time()
        # The usual random number generator using a different seed for each galaxy.
        ud = galsim.UniformDeviate(random_seed+k+1)

        # Choose a random RA, Dec around the sky_center.
        # Note that for this to come out close to a square shape, we need to account for the
        # cos(dec) part of the metric: ds^2 = dr^2 + r^2 d(dec)^2 + r^2 cos^2(dec) d(ra)^2
        # So need to calculate dec first.
        dec = center_dec + (ud()-0.5) * image_size_arcsec * galsim.arcsec
        ra = center_ra + (ud()-0.5) * image_size_arcsec / numpy.cos(dec) * galsim.arcsec
        world_pos = galsim.CelestialCoord(ra,dec)

        # We will need the image position as well, so use the wcs to get that
        image_pos = wcs.toImage(world_pos)

        # We also need this in the tangent plane, which we call "world coordinates" here,
        # since the PowerSpectrum class is really defined on that plane, not in (ra,dec).
        uv_pos = affine.toWorld(image_pos)

        # Get the reduced shears and magnification at this point
        g1, g2, mu = ps.getLensing(pos = uv_pos)

        # Now we will have the COSMOSCatalog make a galaxy profile for us.  It can make either
        # a RealGalaxy using the original HST image and PSF, or a parametric model based on
        # parametric fits to the light distribution of the HST observation.  The parametric
        # models are either a Sersic fit to the data or a bulge + disk fit according to which
        # one gave the better chisq value.  We will select a galaxy at random from the catalog.
        # One could easily do this by choosing an index = int(ud() * cosmos_cat.nobjects), but
        # we will instead allow the catalog to choose a random galaxy for us.  It will remove any
        # selection effects involved in postage stamp creation using weights that are stored in
        # the catalog.  (If for some reason you prefer not to do that, you can always choose a
        # purely random index yourself using int(ud() * cosmos_cat.nobjects).)  We employ this
        # random selection by simply failing to specify an index or identifier for a galaxy, in
        # which case it chooses a random one.

        # First determine whether we will make a real galaxy (`gal_type = 'real'`) or a parametric
        # galaxy (`gal_type = 'parametric'`).  The real galaxies take longer to render, so for this
        # script, we just use them 30% of the time and use parametric galaxies the other 70%.

        # We could just use `ud()<0.3` for this, but instead we introduce another Deviate type
        # available in GalSim that we haven't used yet: BinomialDeviate.
        # It takes an N and p value and returns integers according to a binomial distribution.
        # i.e. How many heads you get after N flips if each flip has a chance, p, of being heads.
        binom = galsim.BinomialDeviate(ud, N=1, p=0.3)
        real = binom()

        if real:
            # For real galaxies, we will want to whiten the noise in the image (below).
            # When whitening the image, we need to make sure the original correlated noise is
            # present throughout the whole image, otherwise the whitening will do the wrong thing
            # to the parts of the image that don't include the original image.  The RealGalaxy
            # stores the correct noise profile to use as the gal.noise attribute.  This noise
            # profile is automatically updated as we shear, dilate, convolve, etc.  But we need to
            # tell it how large to pad with this noise by hand.  This is a bit complicated for the
            # code to figure out on its own, so we have to supply the size for noise padding
            # with the noise_pad_size parameter.

            # The large galaxies will render fine without any noise padding, but the postage stamp
            # for the smaller galaxies will be sized appropriately for the PSF, which may make the
            # stamp larger than the original galaxy image.  The psf image is 40 x 40, although
            # the bright part is much more concentrated than that.  If we pad out the galaxy image
            # to at least 40 x sqrt(2), we should be safe even if the galaxy image is rotated
            # with respect to the psf image.
            #     noise_pad_size = 40 * sqrt(2) * 0.2 arcsec/pixel = 11.3 arcsec
            gal = cosmos_cat.makeGalaxy(gal_type='real', rng=ud, noise_pad_size=11.3)
        else:
            gal = cosmos_cat.makeGalaxy(gal_type='parametric', rng=ud)

        # Apply a random rotation
        theta = ud()*2.0*numpy.pi*galsim.radians
        gal = gal.rotate(theta)

        # Rescale the flux to match our telescope configuration.
        # This automatically scales up the noise variance by flux_scaling**2.
        gal *= flux_scaling

        # Apply the cosmological (reduced) shear and magnification at this position using a single
        # GSObject method.
        gal = gal.lens(g1, g2, mu)

        # Convolve with the PSF.
        final = galsim.Convolve(psf, gal)

        # Account for the fractional part of the position
        # cf. demo9.py for an explanation of this nominal position stuff.
        x_nominal = image_pos.x + 0.5
        y_nominal = image_pos.y + 0.5
        ix_nominal = int(math.floor(x_nominal+0.5))
        iy_nominal = int(math.floor(y_nominal+0.5))
        dx = x_nominal - ix_nominal
        dy = y_nominal - iy_nominal
        offset = galsim.PositionD(dx,dy)

        # We use method='no_pixel' here because the SDSS PSF image that we are using includes the
        # pixel response already.
        stamp = final.drawImage(wcs=wcs.local(image_pos), offset=offset, method='no_pixel')

        # Recenter the stamp at the desired position:
        stamp.setCenter(ix_nominal,iy_nominal)

        # Find the overlapping bounds:
        bounds = stamp.bounds & full_image.bounds

        # Now, if we are using a real galaxy, we want to ether whiten or at least symmetrize the
        # noise on the postage stamp to avoid having to deal with correlated noise in any kind of
        # image processing you would want to do on the final image.  (Like measure galaxy shapes.)

        # Galsim automatically propagates the noise correctly from the initial RealGalaxy object
        # through the applied shear, distortion, rotation, and convolution into the final object's
        # noise attribute.  To make the noise fully white, use the image.whitenNoise() method.
        # The returned value is the variance of the Gaussian noise that is present after the
        # whitening process.

        # However, this is often overkill for many applications.  If it is acceptable to merely end
        # up with noise with some degree of symmetry (say 4-fold or 8-fold symmetry), then you can
        # instead have GalSim just add enough noise to make the resulting noise have this kind of
        # symmetry.  Usually this requires adding significantly less additional noise, which means
        # you can have the resulting total variance be somewhat smaller.  The returned variance
        # corresponds to the zero-lag value of the noise correlation function, which will still have
        # off-diagonal elements.  We can do this step using the image.symmetrizeNoise() method.
        if real:
            if True:
                # We use the symmetrizing option here.
                new_variance = stamp.symmetrizeNoise(final.noise, 8)
            else:
                # Here is how you would do it if you wanted to fully whiten the image.
                new_variance = stamp.whitenNoise(final.noise)

            # We need to keep track of how much variance we have currently in the image, so when
            # we add more noise, we can omit what is already there.
            noise_image[bounds] += new_variance

        # Finally, add the stamp to the full image.
        full_image[bounds] += stamp[bounds]

        time2 = time.time()
        tot_time = time2-time1
        logger.info('Galaxy %d: position relative to center = %s, t=%f s',
                    k, str(uv_pos), tot_time)

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

    # Compute some sky positions of some of the pixels to compare with the values of RA, Dec
    # that ds9 reports.  ds9 always uses (1,1) for the lower left pixel, so the pixel coordinates
    # of these pixels are different by 1, but you can check that the RA and Dec values are
    # the same as what GalSim calculates.
    ra_str = center_ra.hms()
    dec_str = center_dec.dms()
    logger.info('Center of image    is at RA %sh %sm %ss, DEC %sd %sm %ss',
                ra_str[0:3], ra_str[3:5], ra_str[5:], dec_str[0:3], dec_str[3:5], dec_str[5:])
    for (x,y) in [ (0,0), (0,image_size-1), (image_size-1,0), (image_size-1,image_size-1) ]:
        world_pos = wcs.toWorld(galsim.PositionD(x,y))
        ra_str = world_pos.ra.hms()
        dec_str = world_pos.dec.dms()
        logger.info('Pixel (%4d, %4d) is at RA %sh %sm %ss, DEC %sd %sm %ss',x,y,
                    ra_str[0:3], ra_str[3:5], ra_str[5:], dec_str[0:3], dec_str[3:5], dec_str[5:])
    logger.info('ds9 reports these pixels as (1,1), (1,2048), etc. with the same RA, Dec.')


if __name__ == "__main__":
    main(sys.argv)
