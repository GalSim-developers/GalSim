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
- g1,g2 = ps.buildGriddedShears(grid_spacing, ngrid, rng)
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
    grid_g1, grid_g2 = ps.buildGriddedShears(grid_spacing=stamp_size*pixel_scale,
                                             ngrid=n_tiles, rng=rng)

    # Setup the images:
    gal_image = galsim.ImageF(stamp_size * n_tiles , stamp_size * n_tiles)
    psf_image = galsim.ImageF(stamp_size * n_tiles , stamp_size * n_tiles)
    gal_image.setScale(pixel_scale)
    psf_image.setScale(pixel_scale)

    im_center = gal_image.bounds.center()

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

        pos = b.center() - im_center
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

        # Apply the shear from the power spectrum.
        # Note: numpy likes to access values by [iy,ix]
        gal.applyShear(g1 = grid_g1[iy,ix], g2 = grid_g2[iy,ix])

        # Note: another way to access this after having built the g1,g2 grid
        # is to use ps.getShear(pos) which just returns a single shear for that position.
        # The provided position does not have to be on the original grid, but it does
        # need to be contained within the bounds of the full grid. 
        # i.e. only interpolation is allowed -- not extrapolation.
        alt_g1,alt_g2 = ps.getShear(pos)

        # These assert statements demonstrate the the values agree to 1.e-15.
        # (They might not be exactly equal due to numerical rounding errors, but close enough.)
        assert math.fabs(alt_g1 - grid_g1[iy,ix]) < 1.e-15
        assert math.fabs(alt_g2 - grid_g2[iy,ix]) < 1.e-15

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

        # Now determine what we need to do to get our desired S/N
        # See demo5.py for the math behind this calculation.
        sky_level_pix = sky_level * pixel_scale**2
        sn_meas = math.sqrt( numpy.sum(sub_gal_image.array**2) / sky_level_pix )
        flux_scaling = gal_signal_to_noise / sn_meas
        sub_gal_image *= flux_scaling

        # Add Poisson noise -- the CCDNoise can also take another RNG as its argument
        # so it will be part of the same stream of random numbers as ud above.
        sub_gal_image += sky_level_pix
        sub_gal_image.addNoise(galsim.CCDNoise(rng))
        sub_gal_image -= sky_level_pix

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
