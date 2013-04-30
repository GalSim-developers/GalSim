# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
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
test (Nakajima & Bernstein 2007) of 20 each.

New features introduced in this demo:

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

    gal_signal_to_noise = 100       # Great08 "LowNoise" run
    gal_dilation = 3                # Make the galaxies a bit larger than their original size.

    logger.info('Starting demo script 10')
 
    # Read in galaxy catalog
    cat_file_name = 'real_galaxy_catalog_example.fits'
    # This script is designed to be run from the examples directory so dir is a relative path.  
    # But the '../examples/' part lets bin/demo10 also be run from the bin directory.
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

    # Make the galaxies a bit larger than their original observed size.
    for gal in gal_list:
        gal.applyDilation(gal_dilation) 

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
    # The seeds for the objects are random_seed..random_seed+nobj-1 (which comes later), 
    # so use the next one.
    nobj = n_tiles * n_tiles
    rng = galsim.BaseDeviate(random_seed+nobj)

    # Now have the PowerSpectrum object build a grid of shear values for us to use.
    grid_g1, grid_g2 = ps.buildGrid(grid_spacing=stamp_size*pixel_scale, ngrid=n_tiles, rng=rng)

    # Setup the images:
    gal_image = galsim.ImageF(stamp_size * n_tiles , stamp_size * n_tiles)
    psf_image = galsim.ImageF(stamp_size * n_tiles , stamp_size * n_tiles)
    gal_image.setScale(pixel_scale)
    psf_image.setScale(pixel_scale)

    im_center = gal_image.bounds.trueCenter()

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

    # Build each postage stamp:
    for k in range(nobj):
        # The usual random number generator using a different seed for each galaxy.
        rng = galsim.BaseDeviate(random_seed+k)

        # Determine the bounds for this stamp and its center position.
        ix = ix_list[k]
        iy = iy_list[k]
        b = galsim.BoundsI(ix*stamp_size+1 , (ix+1)*stamp_size, 
                           iy*stamp_size+1 , (iy+1)*stamp_size)
        sub_gal_image = gal_image[b]
        sub_psf_image = psf_image[b]

        pos = b.trueCenter() - im_center
        pos = galsim.PositionD(pos.x * pixel_scale , pos.y * pixel_scale)
        # The image comes out as about 211 arcsec across, so we define our variable
        # parameters in terms of (r/100 arcsec), so roughly the scale size of the image.
        r = math.sqrt(pos.x**2 + pos.y**2) / 100
        psf_fwhm = 0.9 + 0.5 * r**2   # arcsec
        psf_e = 0.4 * r**1.5
        psf_beta = (math.atan2(pos.y,pos.x) + math.pi/2) * galsim.radians

        # Define the PSF profile
        psf = galsim.Gaussian(fwhm=psf_fwhm)
        psf.applyShear(e=psf_e, beta=psf_beta)

        # Define the pixel
        pix = galsim.Pixel(pixel_scale)

        # Define the galaxy profile:

        # For this demo, we are doing a ring test where the same galaxy profile is drawn at many
        # orientations stepped uniformly in angle, making a ring in e1-e2 space.
        # We're drawing each profile at 20 different orientations and then skipping to the
        # next galaxy in the list.  So theta steps by 1/20 * 360 degrees:
        theta = k/20. * 360. * galsim.degrees

        # The index needs to increment every 20 objects so we use k/20 using integer math.
        index = k / 20
        gal = gal_list[index]

        # This makes a new copy so we're not changing the object in the gal_list.
        gal = gal.createRotated(theta)

        # Apply the shear from the power spectrum.  We should either turn the gridded shears
        # grid_g1[iy, ix] and grid_g2[iy, ix] into gridded reduced shears using a utility called
        # galsim.lensing.theoryToObserved, or use ps.getShear() which by default gets the reduced
        # shear.  ps.getShear() is also more flexible because it can get the shear at positions that
        # are not on the original grid, as long as they are contained within the bounds of the full
        # grid. So in this example we'll use ps.getShear().
        alt_g1,alt_g2 = ps.getShear(pos)
        gal.applyShear(g1=alt_g1, g2=alt_g2)

        # Apply half-pixel shift in a random direction.
        shift_r = pixel_scale * 0.5
        ud = galsim.UniformDeviate(rng)
        theta = ud() * 2. * math.pi
        dx = shift_r * math.cos(theta)
        dy = shift_r * math.sin(theta)
        gal.applyShift(dx,dy)

        # Make the final image, convolving with psf and pix
        final = galsim.Convolve([psf,pix,gal])

        # Draw the image
        final.draw(sub_gal_image)

        # Now add noise to get our desired S/N
        # See demo5.py for more info about how this works.
        sky_level_pixel = sky_level * pixel_scale**2
        noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)
        sub_gal_image.addNoiseSNR(noise, gal_signal_to_noise)

        # Draw the PSF image:
        # We use real space convolution to avoid some of the 
        # artifacts that can show up with Fourier convolution.
        # The level of the artifacts is quite low, but when drawing with
        # no noise, they are apparent with ds9's zscale viewing.
        final_psf = galsim.Convolve([psf,pix], real_space=True)

        # For the PSF image, we also shift the PSF by the same amount.
        final_psf.applyShift(dx,dy)

        # No noise on PSF images.  Just draw it as is.
        final_psf.draw(sub_psf_image)

        logger.info('Galaxy (%d,%d): position relative to center = %s', ix,iy,str(pos))

    logger.info('Done making images of postage stamps')

    # Now write the images to disk.
    images = [ gal_image , psf_image ]
    galsim.fits.writeMulti(images, file_name)
    logger.info('Wrote image to %r',file_name) 

if __name__ == "__main__":
    main(sys.argv)
