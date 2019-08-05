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
Demo #10

The tenth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script uses both a variable PSF and variable shear, taken from a power spectrum, along
the lines of a Great10 (Kitching, et al, 2012) image.  The galaxies are placed on a grid
(10 x 10 in this case, rather than 100 x 100 in the interest of time.)  Each postage stamp
is 48 x 48 pixels.  Instead of putting the PSF images on a separate image, we package them
as the second HDU in the file.  For the galaxies, we use a random selection from 5 specific
RealGalaxy objects, selected to be 5 particularly irregular ones. (These are taken from
the same catalog of 100 objects that demo6 used.)  The galaxies are oriented in a ring
test (Nakajima & Bernstein 2007) of 20 each.  And we again output a truth catalog with the
correct applied shear for each object (among other information).

New features introduced in this demo:

- im.wcs = galsim.OffsetWCS(scale, origin)
- rng = galsim.BaseDeviate(seed)
- obj = galsim.RealGalaxy(real_galaxy_catalog, id)
- obj = galsim.Convolve([list], real_space)
- ps = galsim.PowerSpectrum(e_power_function, b_power_function)
- g1,g2 = ps.buildGrid(grid_spacing, ngrid, rng)
- g1,g2 = ps.getShear(pos)
- galsim.random.permute(rng, list1, list2, ...)

- Choosing PSF parameters as a function of (x,y)
- Selecting RealGalaxy by ID rather than index.
- Putting the PSF image in a second HDU in the same file as the main image.
- Using PowerSpectrum for the applied shear.
- Doing a full ring test (i.e. not just 90 degree rotated pairs)
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
      - The second HDU has the corresponding PSF image.
      - Applied shear is from a power spectrum P(k) ~ k^1.8.
      - Galaxies are real galaxies oriented in a ring test of 20 each.
      - The PSF is Gaussian with FWHM, ellipticity and position angle functions of (x,y)
      - Noise is Poisson using a nominal sky value of 1.e6.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo10")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    n_tiles = 10                    # number of tiles in each direction.
    stamp_size = 48                 # pixels

    pixel_scale = 0.44              # arcsec / pixel
    sky_level = 1.e6                # ADU / arcsec^2

    # The random seed is used for both the power spectrum realization and the random properties
    # of the galaxies.
    random_seed = 3339201

    # Make output directory if not already present.
    if not os.path.isdir('output'):
        os.mkdir('output')

    file_name = os.path.join('output','power_spectrum.fits')

    # These will be created for each object below.  The values we'll use will be functions
    # of (x,y) relative to the center of the image.  (r = sqrt(x^2+y^2))
    # psf_fwhm = 0.9 + 0.5 * (r/100)^2  -- arcsec
    # psf_e = 0.4 * (r/100)^1.5         -- large value at the edge, so visible by eye.
    # psf_beta = atan2(y/x) + pi/2      -- tangential pattern

    gal_dilation = 3                # Make the galaxies a bit larger than their original size.
    gal_signal_to_noise = 100       # Pretty high.
    psf_signal_to_noise = 1000      # Even higher.

    logger.info('Starting demo script 10')

    # Read in galaxy catalog
    cat_file_name = 'real_galaxy_catalog_23.5_example.fits'
    dir = 'data'
    real_galaxy_catalog = galsim.RealGalaxyCatalog(cat_file_name, dir=dir)
    logger.info('Read in %d real galaxies from catalog', real_galaxy_catalog.nobjects)

    # List of IDs to use.  We select 5 particularly irregular galaxies for this demo.
    # Then we'll choose randomly from this list.
    id_list = [ 106416, 106731, 108402, 116045, 116448 ]

    # Make the 5 galaxies we're going to use here rather than remake them each time.
    # This means the Fourier transforms of the real galaxy images don't need to be recalculated
    # each time, so it's a bit more efficient.
    gal_list = [ galsim.RealGalaxy(real_galaxy_catalog, id=id) for id in id_list ]
    # Grab the index numbers before we transform them and lose the index attribute.
    cosmos_index = [ gal.index for gal in gal_list ]

    # Make the galaxies a bit larger than their original observed size.
    gal_list = [ gal.dilate(gal_dilation) for gal in gal_list ]

    # Setup the PowerSpectrum object we'll be using:
    ps = galsim.PowerSpectrum(lambda k : k**1.8)
    # The argument here is "e_power_function" which defines the E-mode power to use.

    # There is also a b_power_function if you want to include any B-mode power:
    #     ps = galsim.PowerSpectrum(e_power_function, b_power_function)

    # You may even omit the e_power_function argument and have a pure B-mode power spectrum.
    #     ps = galsim.PowerSpectrum(b_power_function = b_power_function)


    # All the random number generator classes derive from BaseDeviate.
    # When we construct another kind of deviate class from any other
    # kind of deviate class, the two share the same underlying random number
    # generator.  Sometimes it can be clearer to just construct a BaseDeviate
    # explicitly and then construct anything else you need from that.
    # Note: A BaseDeviate cannot be used to generate any values.  It can
    # only be used in the constructor for other kinds of deviates.
    # The seeds for the objects are random_seed+1..random_seed+nobj.
    # The seeds for things at the image or file level use random_seed itself.
    nobj = n_tiles * n_tiles
    rng = galsim.BaseDeviate(random_seed)

    # Have the PowerSpectrum object build a grid of shear values for us to use.
    grid_g1, grid_g2 = ps.buildGrid(grid_spacing=stamp_size*pixel_scale, ngrid=n_tiles+1, rng=rng)

    # Setup the images:
    gal_image = galsim.ImageF(stamp_size * n_tiles , stamp_size * n_tiles)
    psf_image = galsim.ImageF(stamp_size * n_tiles , stamp_size * n_tiles)

    # Update the image WCS to use the image center as the origin of the WCS.
    # The class that acts like a PixelScale except for this offset is called OffsetWCS.
    im_center = gal_image.true_center
    wcs = galsim.OffsetWCS(scale=pixel_scale, origin=im_center)
    gal_image.wcs = wcs
    psf_image.wcs = wcs

    # We will place the tiles in a random order.  To do this, we make two lists for the
    # ix and iy values.  Then we apply a random permutation to the lists (in tandem).
    ix_list = []
    iy_list = []
    for ix in range(n_tiles):
        for iy in range(n_tiles):
            ix_list.append(ix)
            iy_list.append(iy)
    # This next function will use the given random number generator, rng, and use it to
    # randomly permute any number of lists.  All lists will have the same random permutation
    # applied.
    galsim.random.permute(rng, ix_list, iy_list)

    # Initialize the OutputCatalog for the truth values
    names = [ 'gal_num', 'x_image', 'y_image',
              'psf_e1', 'psf_e2', 'psf_fwhm',
              'cosmos_id', 'cosmos_index', 'theta',
              'g1', 'g2', 'shift_x', 'shift_y' ]
    types = [ int, float, float,
              float, float, float,
              str, int, float,
              float, float, float, float ]
    truth_catalog = galsim.OutputCatalog(names, types)

    # Build each postage stamp:
    for k in range(nobj):
        # The usual random number generator using a different seed for each galaxy.
        rng = galsim.BaseDeviate(random_seed+k+1)

        # Determine the bounds for this stamp and its center position.
        ix = ix_list[k]
        iy = iy_list[k]
        b = galsim.BoundsI(ix*stamp_size+1 , (ix+1)*stamp_size,
                           iy*stamp_size+1 , (iy+1)*stamp_size)
        sub_gal_image = gal_image[b]
        sub_psf_image = psf_image[b]

        pos = wcs.toWorld(b.true_center)
        # The image comes out as about 211 arcsec across, so we define our variable
        # parameters in terms of (r/100 arcsec), so roughly the scale size of the image.
        rsq = (pos.x**2 + pos.y**2)
        r = math.sqrt(rsq)

        psf_fwhm = 0.9 + 0.5 * rsq / 100**2   # arcsec
        psf_e = 0.4 * (r/100.)**1.5
        psf_beta = (math.atan2(pos.y,pos.x) + math.pi/2) * galsim.radians

        # Define the PSF profile
        psf = galsim.Gaussian(fwhm=psf_fwhm)
        psf_shape = galsim.Shear(e=psf_e, beta=psf_beta)
        psf = psf.shear(psf_shape)

        # Define the galaxy profile:

        # For this demo, we are doing a ring test where the same galaxy profile is drawn at many
        # orientations stepped uniformly in angle, making a ring in e1-e2 space.
        # We're drawing each profile at 20 different orientations and then skipping to the
        # next galaxy in the list.  So theta steps by 1/20 * 360 degrees:
        theta_deg = (k%20) * 360. / 20
        theta = theta_deg * galsim.degrees

        # The index needs to increment every 20 objects so we use k/20 using integer math.
        index = k // 20
        gal = gal_list[index]

        # This makes a new copy so we're not changing the object in the gal_list.
        gal = gal.rotate(theta)

        # Apply the shear from the power spectrum.  We should either turn the gridded shears
        # grid_g1[iy, ix] and grid_g2[iy, ix] into gridded reduced shears using a utility called
        # galsim.lensing.theoryToObserved, or use ps.getShear() which by default gets the reduced
        # shear.  ps.getShear() is also more flexible because it can get the shear at positions that
        # are not on the original grid, as long as they are contained within the bounds of the full
        # grid. So in this example we'll use ps.getShear().
        alt_g1,alt_g2 = ps.getShear(pos)
        gal = gal.shear(g1=alt_g1, g2=alt_g2)

        # Apply half-pixel shift in a random direction.
        shift_r = pixel_scale * 0.5
        ud = galsim.UniformDeviate(rng)
        t = ud() * 2. * math.pi
        dx = shift_r * math.cos(t)
        dy = shift_r * math.sin(t)
        gal = gal.shift(dx,dy)

        # Make the final image, convolving with the psf
        final = galsim.Convolve([psf,gal])

        # Draw the image
        final.drawImage(sub_gal_image)

        # For the PSF image, we don't match the galaxy shift.  Rather, we use the offset
        # parameter to drawImage to apply a random offset of up to 0.5 pixels in each direction.
        # Note the difference in units between shift and offset.  The shift is applied to the
        # surface brightness profile, so it is in sky coordinates (as all dimension are for
        # GSObjects), which are arcsec here.  The offset though is applied to the image itself,
        # so it is in pixels.  Hence, we don't multiply by pixel_scale.
        psf_dx = ud() - 0.5
        psf_dy = ud() - 0.5
        psf_offset = galsim.PositionD(psf_dx, psf_dy)

        # Draw the PSF image:
        # We use real space integration over the pixels to avoid some of the
        # artifacts that can show up with Fourier convolution.
        # The level of the artifacts is quite low, but when drawing with
        # so little noise, they are apparent with ds9's zscale viewing.
        psf.drawImage(sub_psf_image, method='real_space', offset=psf_offset)

        # Build the noise model: Poisson noise with a given sky level.
        sky_level_pixel = sky_level * pixel_scale**2
        noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)

        # Add noise to the PSF image, using the normal noise model, but scaling the
        # PSF flux high enough to reach the desired signal-to-noise.
        # See demo5.py for more info about how this works.
        sub_psf_image.addNoiseSNR(noise, psf_signal_to_noise)

        # And also to the galaxy image using its signal-to-noise.
        sub_gal_image.addNoiseSNR(noise, gal_signal_to_noise)

        # Add the truth values to the truth catalog
        row = [ k, b.true_center.x, b.true_center.y,
                psf_shape.e1, psf_shape.e2, psf_fwhm,
                id_list[index], cosmos_index[index], (theta_deg % 360.),
                alt_g1, alt_g2, dx, dy ]
        truth_catalog.addRow(row)

        logger.info('Galaxy (%d,%d): position relative to center = %s', ix,iy,str(pos))

    logger.info('Done making images of postage stamps')

    # In this case, we'll attach the truth catalog as an additional HDU in the same file as
    # the image data.
    truth_hdu = truth_catalog.writeFitsHdu()

    # Now write the images to disk.
    images = [ gal_image , psf_image, truth_hdu ]
    # Any items in the "images" list that is already an hdu is just used directly.
    # The actual images are converted to FITS hdus that contain the image data.
    galsim.fits.writeMulti(images, file_name)
    logger.info('Wrote image to %r',file_name)

if __name__ == "__main__":
    main(sys.argv)
