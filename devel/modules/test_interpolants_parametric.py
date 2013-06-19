"""@file test_interpolants_parametric.py  Tests of interpolants using parametric galaxy models.

A companion script to `test_interpolants.py`, but instead of using `RealGalaxy` objects we instead
use Sersic models drawn into `InterpolatedImage` instances to try and get to the nitty-gritty of
the issues with interpolators.

The parameters of the Sersic images come from a COSMOS best-fitting Sersic model catalog.
"""

import numpy as np
import galsim
import test_interpolants

SERSIC_IMAGE_SIZE = 512 # For initial image of the Sersic at Hubble resolution, make nice and large
TEST_IMAGE_SIZE = SERSIC_IMAGE_SIZE  # For speed could make this smaller

def calculate_interpolated_image_g1g2sigma(images, psf=None, dx_input=None, dx_test=None, 
                                           shear=None, magnification=None, angle=None, shift=None, 
                                           x_interpolant=None, k_interpolant=None, padding=None):
    """Takes a list of drawn images of Sersic galaxies, reads them into an InterpolatedImage object
    using the supplied parameters, and calculates the g1, g2 and sigma of the output.
    """
    # Some input parsing
    if padding is None:
        pad_factor = 0.
    else:
        pad_factor = padding
    # Loop over the images and generate an InterpolatedImage from the pixel values
    for sersic_image in sersic_images:

        # Build the raw InterpolatedImage
        test_gal = galsim.InterpolatedImage(
            sersic_image, dx=dx, x_interpolant=x_interpolant, k_interpolant=k_interpolant,
            pad_factor=pad_factor)
        # Apply shears, magnification, rotation and shifts if requested
        if shear is not None:
            test_gal.applyShear(g1=shear[0], g2=shear[1])
        if magnification is not None:
            test_gal.applyMagnification(magnification)
        if angle is not None:
            if not isinstance(angle, galsim.Angle):
                raise ValueError("Input kwarg angle must be a galsim.Angle instance.")
            test_gal.applyRotation(angle)
        if shift is not None:
            if not isinstance(shift, galsim.PositionD):
                raise ValueError("Input kwarg shift must be a galsim.PositionD instance.")
            test_gal.applyShift(shift) 
        # Apply a PSF if requested
        if psf is not None:
            test_final = galsim.Convolve([test_gal, psf])
        else:
            test_final = test_gal
        # Draw into the test image and calculate adaptive moments
        test_image = galsim.ImageD(TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)
        test_image.setScale(dx_output)

        results = test_interpolants.CatchAdaptiveMomErrors(


def draw_sersic_images(narr, hlrarr, gobsarray, random_seed=None, nmin=0.3, nmax=4.2,
                       image_size=512, pixel_scale=0.03):
    """Given input NumPy arrays of Sersic n, half light radius, and |g|, draw a list of Sersic
    images with n values within range, at random orientations.
    """
    # Initialize the random number generator
    if random_seed is None:
        u = galsim.UniformDeviate()
    else:
        u = galsim.UniformDeviate(random_seed)

    # Loop over all the input params and make an sersic galaxy image from each
    sersic_images = []
    print "Drawing Sersic images"
    for n, hlr, gobs in zip(narr, hlrarr, gobsarr):

        # First check we are only using Sersic indices in range
        if n <= nmin or n >= nmax: continue # This goes to the next iteration

        # Otherwise set up the image to draw our COSMOS sersic profiles into
        sersic_image = galsim.ImageD(image_size, image_size)
        sersic_image.setScale(pixel_scale)
        # Build the galaxy
        galaxy = galsim.Sersic(n=n, half_light_radius=hlr)
        # Apply the ellipticity of the correct magnitude with a random rotation
        theta_rot = 2. * np.pi * u() # Random orientation
        galaxy.applyShear(g1=gobs*np.cos(2.*theta_rot), g2=gobs*np.sin(2.*theta_rot))
        galaxy.draw(sersic_image, dx=test_interpolants.space_pixel_scale)
        sersic_images.append(sersic_image)

    # Return this list of drawn images
    return sersic_images


if __name__ == "__main__":

    # Import the Sersic galaxy sample module
    try:
        import galaxy_sample
    except ImportError:
        import sys
        sys.path.append('../external/test_sersic_highn')
        import galaxy_sample

    # Get the COSMOS galaxy sample parameters
    ns_cosmos, hlrs_cosmos, gobss_cosmos = galaxy_sample.get()
    # Only use the first test_interpolants.nitems galaxies in these lists, starting at
    # test_interpolants.first_index
    istart = test_interpolants.first_index
    iend = istart + test_interpolants.nitems
    ns_cosmos = ns_cosmos[istart: iend]
    hlrs_cosmos = hlrs_cosmos[istart: iend]
    gobss_cosmos = gobss_cosmos[istart: iend]

    # Draw a whole load of images of Sersic profiles at random orientations using these params
    sersic_images = draw_sersic_images(
        ns_cosmos, hlrs_cosmos, gobss_cosmos, random_seed=RANDOM_SEED, nmin=0.3, nmax=4.2,
        image_size=IMAGE_SIZE, pixel_scale=test_interpolants.space_pixel_scale)

    # Let's just do space and ground-based PSFs, and define a tuple storing these to iterate over
    psfs = (
        galsim.Airy(lam_over_diam=test_interpolants.space_lam_over_diam),
        galsim.Convolve(
            galsim.Kolmogorov(fwhm=test_interpolants.ground_fwhm),
            galsim.Airy(lam_over_diam=test_interpolants.ground_lam_over_diam)
        )
    )

    # Then we start the grand loop producing output in a similar fashion to test_interpolants.py
    for psf in psfs:

        for padding in test_interpolants.padding_list:

            for interpolant in test_interpolants.use_interpolants:


