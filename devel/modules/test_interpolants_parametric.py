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

"""@file test_interpolants_parametric.py  Tests of interpolants using parametric galaxy models.

A companion script to `test_interpolants.py`, but instead of using `RealGalaxy` objects we instead
use Sersic models drawn into `InterpolatedImage` instances to try and get to the nitty-gritty of
the issues with interpolators.

The parameters of the Sersic images come from a COSMOS best-fitting Sersic model catalog.
"""

import cPickle
import numpy as np
import galsim
import test_interpolants


SERSIC_IMAGE_SIZE = 512 # For initial image of the Sersic at Hubble resolution, make nice and large
TEST_IMAGE_SIZE = SERSIC_IMAGE_SIZE  # For speed could make this smaller
# Dictionary for parsing the test_interpolants.interpolant_list into galsim Interpolants 
INTERPOLANT_DICT = {
    "nearest" : galsim.Nearest(),
    "sinc" : galsim.SincInterpolant(),
    "linear" : galsim.Linear(),
    "cubic" : galsim.Cubic(),
    "quintic" : galsim.Quintic(), 
    "lanczos3" : galsim.Lanczos(3),
    "lanczos4" : galsim.Lanczos(4),
    "lanczos5" : galsim.Lanczos(5),
    "lanczos7" : galsim.Lanczos(7)}

# Output filenames
DELTA_FILENAME = 'interpolant_test_parametric_output_delta.dat'
ORIGINAL_FILENAME = 'interpolant_test_parametric_output_original.dat'

NITEMS = 300 # more expensive but we beat down the errors this way

LAM_OVER_DIAM_COSMOS = 814.e-9 / 2.4 # All the original images in Melanie's tests were from COSMOS
                                     # F814W, so this is a crude approximation to the PSF scale in
                                     # radians, ~0.07 arcsec
COSMOS_PSF = galsim.Airy(lam_over_diam=LAM_OVER_DIAM_COSMOS * 180. * 3600. / np.pi)

class InterpolationDataNoConfig:
    """Quick container class for passing around data from these tests, but not using config.
    """ 
    def __init__(self, g1obs=None, g2obs=None, sigmaobs=None, err_g1obs=None, err_g2obs=None, 
                 err_sigmaobs=None, dx_input=test_interpolants.pixel_scale,
                 dx_test=test_interpolants.pixel_scale, shear=None, magnification=None,
                 angle=None, shift=None, x_interpolant=None, k_interpolant=None, padding=None,
                 image_type='delta'):
        self.g1obs = g1obs
        self.g2obs = g2obs
        self.sigmaobs = sigmaobs
        self.err_g1obs = err_g1obs
        self.err_g2obs = err_g2obs
        self.err_sigmaobs = err_sigmaobs
        self.dx_input = dx_input
        self.dx_test = dx_test
        self.test_image_size = TEST_IMAGE_SIZE
        # Parse and store default/non-default shear, mag, rotation and shift
        if shear is None:
            self.shear = [0., 0.]
        else:
            self.shear = shear
        if magnification is None:
            self.magnification = 1.
        else:
            self.magnification = magnification
        if angle is None:
            self.angle = 0.
        else:
            self.angle = angle.rad * 180. / np.pi
        if shift is None:
            self.shiftx = 0.
            self.shifty = 0.
        else:
            self.shiftx = shift.x
            self.shifty = shift.y
        # Store the interpolants
        if x_interpolant is None:
            self.x_interpolant = 'default'
        else:  
            self.x_interpolant = x_interpolant
        if k_interpolant is None:
            self.k_interpolant = 'default'
        else:  
            self.k_interpolant = k_interpolant
        # Store padding & image type
        if padding is None:
            self.padding = 0
        else:
            self.padding = padding
        self.image_type = image_type


def calculate_interpolated_image_g1g2sigma(images, psf=None, dx_input=None, dx_test=None, 
                                           shear=None, magnification=None, angle=None, shift=None, 
                                           x_interpolant=None, k_interpolant=None, padding=None,
                                           image_type='delta'):
    """Takes a list of drawn images of Sersic galaxies, reads them into an InterpolatedImage object
    using the supplied parameters, and calculates the g1, g2 and sigma of the output.
    """
    # Some input parsing
    if padding is None:
        pad_factor = 0.
    else:
        pad_factor = padding
    # Loop over the images and generate an InterpolatedImage from the pixel values
    g1obs_list = []
    g2obs_list = []
    sigmaobs_list = []
    # Parse input interpolants
    if x_interpolant is not None:
        x_interpolant_obj = INTERPOLANT_DICT[x_interpolant]
    else:
        x_interpolant_obj = None
    if k_interpolant is not None:
        k_interpolant_obj = INTERPOLANT_DICT[k_interpolant]
    else:
        k_interpolant_obj = None
    # Loop over images
    for image in images:

        # Build the raw InterpolatedImage
        test_gal = galsim.InterpolatedImage(
            image, scale=dx_input, x_interpolant=x_interpolant_obj, k_interpolant=k_interpolant_obj,
            pad_factor=pad_factor)
        # Apply shears, magnification, rotation and shifts if requested
        if shear is not None:
            test_gal = test_gal.shear(g1=shear[0], g2=shear[1])
        if magnification is not None:
            test_gal = test_gal.magnify(magnification)
        if angle is not None:
            if not isinstance(angle, galsim.Angle):
                raise ValueError("Input kwarg angle must be a galsim.Angle instance.")
            test_gal = test_gal.rotate(angle)
        if shift is not None:
            if not isinstance(shift, galsim.PositionD):
                raise ValueError("Input kwarg shift must be a galsim.PositionD instance.")
            test_gal = test_gal.shift( # Shifts are in pixel units so convert to arcsec
                dx=shift.x*dx_test, dy=shift.y*dx_test) 
        # Apply a PSF if requested
        if psf is not None:
            test_final = galsim.Convolve([test_gal, psf])
        else:
            test_final = test_gal
        # Draw into the test image and calculate adaptive moments
        test_image = galsim.ImageD(TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)
        test_image.scale = dx_test
        test_final.drawImage(test_image, method='no_pixel', scale=dx_test)
        trial_result = test_interpolants.CatchAdaptiveMomErrors(test_image)
        if isinstance(trial_result, float):
            g1obs_list.append(-10)
            g2obs_list.append(-10)
            sigmaobs_list.append(-10)
        elif isinstance(trial_result, galsim.hsm.ShapeData):
            g1obs_list.append(trial_result.observed_shape.g1)
            g2obs_list.append(trial_result.observed_shape.g2)
            sigmaobs_list.append(trial_result.moments_sigma)
        else:
            raise TypeError("Unexpected output from test_interpolants.CatchAdaptiveMomErrors().")
    # Return a container with the results
    ret = InterpolationDataNoConfig(
        g1obs=g1obs_list, g2obs=g2obs_list, sigmaobs=sigmaobs_list, dx_input=dx_input,
        dx_test=dx_test, shear=shear, magnification=magnification, angle=angle, shift=shift,  
        x_interpolant=x_interpolant, k_interpolant=k_interpolant, padding=padding,
        image_type=image_type)
    return ret

def draw_sersic_images(narr, hlrarr, gobsarr, random_seed=None, nmin=0.3, nmax=4.2,
                       image_size=SERSIC_IMAGE_SIZE, pixel_scale=test_interpolants.pixel_scale,
                       psf=COSMOS_PSF):
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
        sersic_image.scale = pixel_scale
        # Build the galaxy
        galaxy = galsim.Sersic(n=n, half_light_radius=hlr,
                               gsparams=galsim.GSParams(maximum_fft_size=8192))
        # Apply the ellipticity of the correct magnitude with a random rotation
        theta_rot = 2. * np.pi * u() # Random orientation
        galaxy = galaxy.shear(g1=gobs*np.cos(2.*theta_rot), g2=gobs*np.sin(2.*theta_rot))
        if psf is None:
            final = galaxy
        elif isinstance(psf, galsim.GSObject):
            final = galsim.Convolve([galaxy, psf])
        else:
            raise TypeError("Input psf kwarg must be a GSObject or NoneType.") 
        final.drawImage(sersic_image, method='no_pixel', scale=pixel_scale)
        sersic_images.append(sersic_image)

    # Return this list of drawn images
    return sersic_images

def run_tests(use_interpolants, nitems=test_interpolants.default_nitems):
    """Run the tests for the specified interpolants."""

    import sys
    # Import the Sersic galaxy sample module
    try:
        import galaxy_sample
    except ImportError:
        import sys
        sys.path.append('../external/test_sersic_highn')
        import galaxy_sample

    # Get the COSMOS galaxy sample parameters
    ns_cosmos, hlrs_cosmos, gobss_cosmos = galaxy_sample.get()
    # Only use the first nitems galaxies in these lists, starting at test_interpolants.first_index
    istart = test_interpolants.default_first_index
    iend = istart + nitems
    ns_cosmos = ns_cosmos[istart: iend]
    hlrs_cosmos = hlrs_cosmos[istart: iend]
    gobss_cosmos = gobss_cosmos[istart: iend]

    # Draw a whole load of images of Sersic profiles at random orientations using these params
    sersic_images = draw_sersic_images(
        ns_cosmos, hlrs_cosmos, gobss_cosmos, random_seed=test_interpolants.rseed, nmin=0.3,
        nmax=4.2, image_size=SERSIC_IMAGE_SIZE, pixel_scale=test_interpolants.pixel_scale)

    # Calculate the reference results for g1obs, g2obs and sigma for these reference images
    g1_list = []
    g2_list = []
    sigma_list = []
    print "Calculating reference g1, g2 & sigma for "+str(len(sersic_images))+" Sersic images"
    for sersic_image in sersic_images:
        shape = test_interpolants.CatchAdaptiveMomErrors(sersic_image)
        if isinstance(shape, float):
            g1_list.append(-10)
            g2_list.append(-10)
            sigma_list.append(-10)
        elif isinstance(shape, galsim.hsm.ShapeData):
            g1_list.append(shape.observed_shape.g1)
            g2_list.append(shape.observed_shape.g2)
            sigma_list.append(shape.moments_sigma)
        else:
            raise TypeError("Unexpected output from test_interpolants.CatchAdaptiveMomErrors().")
    g1_list = np.asarray(g1_list)
    g2_list = np.asarray(g2_list)
    sigma_list = np.asarray(sigma_list)

    # Then start the interpolant tests...
    # Define a dict storing PSFs to iterate over along with the appropriate test pixel scale and
    # filename
    psf_dict = {
        "delta" : (
            galsim.Gaussian(1.e-8), test_interpolants.pixel_scale, DELTA_FILENAME),
        "original" : (
            None, test_interpolants.pixel_scale, ORIGINAL_FILENAME),
    }
    print''
    # Then we start the grand loop producing output in a similar fashion to test_interpolants.py
    for image_type in ("delta", "original"):
 
        # Get the correct PSF and test image pixel scale
        psf = psf_dict[image_type][0]
        dx_test = psf_dict[image_type][1]
        outfile = open(psf_dict[image_type][2], 'wb')
        print "Writing test results to "+str(outfile)
        for padding in test_interpolants.padding_list:

            print "Using padding = "+str(padding)
            for interpolant in use_interpolants:

                print "Using interpolant: "+str(interpolant)
                print 'Running Angle tests'
                for angle in test_interpolants.angle_list: # Possible rotation angles

                    sys.stdout.write('.')
                    sys.stdout.flush()
                    dataXint = calculate_interpolated_image_g1g2sigma(
                        sersic_images, psf=psf, dx_input=test_interpolants.pixel_scale,
                        dx_test=dx_test, shear=None, magnification=None, angle=angle*galsim.degrees,
                        shift=None, x_interpolant=interpolant, padding=padding,
                        image_type=image_type)
                    test_interpolants.print_results(
                        outfile, g1_list, g2_list, sigma_list, dataXint)
                    dataKint = calculate_interpolated_image_g1g2sigma(
                        sersic_images, psf=psf, dx_input=test_interpolants.pixel_scale,
                        dx_test=dx_test, shear=None, magnification=None, angle=angle*galsim.degrees,
                        shift=None, k_interpolant=interpolant, padding=padding,
                        image_type=image_type)
                    test_interpolants.print_results(
                        outfile, g1_list, g2_list, sigma_list, dataKint)
                sys.stdout.write('\n')
 
                print 'Running Shear/Magnification tests'
                for (g1, g2, mag) in test_interpolants.shear_and_magnification_list:

                    sys.stdout.write('.')
                    sys.stdout.flush()
                    dataXint = calculate_interpolated_image_g1g2sigma(
                        sersic_images, psf=psf, dx_input=test_interpolants.pixel_scale,
                        dx_test=dx_test, shear=(g1, g2), magnification=mag, angle=None,
                        shift=None, x_interpolant=interpolant, padding=padding,
                        image_type=image_type)
                    test_interpolants.print_results(
                        outfile, g1_list, g2_list, sigma_list, dataXint)
                    dataKint = calculate_interpolated_image_g1g2sigma(
                        sersic_images, psf=psf, dx_input=test_interpolants.pixel_scale,
                        dx_test=dx_test, shear=(g1, g2), magnification=mag, angle=None,
                        shift=None, k_interpolant=interpolant, padding=padding,
                        image_type=image_type)
                    test_interpolants.print_results(
                        outfile, g1_list, g2_list, sigma_list, dataKint)
                sys.stdout.write('\n')

                print 'Running Shift tests'
                for shift in test_interpolants.shift_list:

                    sys.stdout.write('.')
                    sys.stdout.flush()
                    dataXint = calculate_interpolated_image_g1g2sigma(
                        sersic_images, psf=psf, dx_input=test_interpolants.pixel_scale,
                        dx_test=dx_test, shear=None, magnification=None, angle=None,
                        shift=shift, x_interpolant=interpolant, padding=padding,
                        image_type=image_type)
                    test_interpolants.print_results(
                        outfile, g1_list, g2_list, sigma_list, dataXint)
                    dataKint = calculate_interpolated_image_g1g2sigma(
                        sersic_images, psf=psf, dx_input=test_interpolants.pixel_scale,
                        dx_test=dx_test, shear=None, magnification=None, angle=None,
                        shift=shift, k_interpolant=interpolant, padding=padding,
                        image_type=image_type)
                    test_interpolants.print_results(
                        outfile, g1_list, g2_list, sigma_list, dataKint)
                sys.stdout.write('\n')

                print ''

        print "Finished tests for image_type: "+str(image_type) 
        print ""
        outfile.close()


if __name__ == "__main__":

    use_interpolants = test_interpolants.interpolant_list[2:]
    run_tests(use_interpolants, nitems=NITEMS)

