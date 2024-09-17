# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

from copy import deepcopy
import numpy as np

import galsim
from galsim_test_helpers import *


# Save images used in regression testing for manual inspection?
save_profiles = False

# set up any necessary info for tests
# Note that changes here should match changes to test image files
image_dir = './inclined_exponential_images'

# Values here are strings, so the filenames will be sure to work (without truncating zeros)

# Tuples are in the order: (flux, n, inclination angle (radians), scale radius, scale height,
#                           truncation factor, position angle)

# Parameter sets valid for the inclined exponential case
inclined_exponential_test_parameters = (
    ("1.0", "1.0", "0.0" , "3.0", "0.3", "0.0", "0.0"),
    ("10.0", "1.0", "1.3" , "3.0", "0.5", "0.0", "0.0"),
    ("0.1", "1.0", "0.2" , "3.0", "0.5", "0.0", "0.0"),
    ("1.0", "1.0", "0.01", "3.0", "0.5", "0.0", "0.0"),
    ("2.1", "1.0", "1.57", "2.5", "0.2", "0.0", "7.4"),
    ("1.0", "1.0", "0.1" , "2.0", "1.0", "0.0", "-0.2"),
    ("1.0", "1.0", "0.78", "2.0", "0.5", "0.0", "-0.2"),)

# Parameter sets valid only for the inclined Sersic case
inclined_sersic_test_parameters = (
    ("2.0", "1.5", "0.0" , "3.0", "0.3", "0.0", "0.0"),
    ("1.5", "2.0", "0.0" , "3.0", "0.3", "0.0", "0.0"),
    ("1.1", "0.5", "0.01" , "3.0", "0.3", "0.0", "0.0"),
    ("10.0", "1.0", "0.1" , "1.9", "0.3", "4.5", "7.3"),
    ("1.0e6", "1.5", "0.2" , "2.1", "0.2", "3.9", "-0.9"),
    ("1.0e-6", "2.0", "0.3" , "3.4", "0.1", "5.0", "0.6"),
    ("2.3e4", "0.5", "1.57" , "1.8", "0.5", "2.5", "0.3"),)

# Parameter sets used for regression tests of Sersic profiles
inclined_sersic_regression_test_parameters = (
    ("1.0", "0.5", "0.7", "4.0", "0.2", "0.0", "0.5"),
    ("1.0", "0.8", "1.4", "2.5", "0.1", "4.0", "1.1"),
    ("1.0", "1.2", "2.0", "1.5", "0.5", "6.5", "5.5"),
    ("1.0", "1.5", "0.3", "3.0", "0.3", "0.0", "0.0"),)

image_nx = 64
image_ny = 64

def get_prof(mode, *args, **kwargs):
    """Function to get either InclinedExponential or InclinedSersic (with n=1, trunc=0)
       depending on mode
    """
    new_kwargs = deepcopy(kwargs)
    if len(args) > 0:
        new_kwargs["inclination"] = args[0]
    if len(args) > 1:
        new_kwargs["scale_radius"] = args[1]
    if len(args) > 2:
        new_kwargs["scale_height"] = args[2]

    if mode == "InclinedSersic":

        if not "trunc" in new_kwargs:
            new_kwargs["trunc"] = 0.
        if not "n" in new_kwargs:
            new_kwargs["n"] = 1.

        prof = galsim.InclinedSersic(**new_kwargs)
    else:
        if "trunc" in new_kwargs:
            del new_kwargs["trunc"]
        if "n" in new_kwargs:
            del new_kwargs["n"]
        if "flux_untruncated" in new_kwargs:
            del new_kwargs["flux_untruncated"]
        prof = galsim.InclinedExponential(**new_kwargs)

    return prof


@timer
def test_regression():
    """Test that the inclined exponential profile matches the results from Lance Miller's code.
       Reference images are provided in the ./inclined_exponential_images directory, as well as
       the code ('hankelcode.c') used to generate them."""

    for mode in ("InclinedExponential", "InclinedSersic"):

        for (flux, _sersic_n, inc_angle, scale_radius, scale_height,
             _trunc_factor, pos_angle) in inclined_exponential_test_parameters:

            image_filename = "galaxy_" + inc_angle + "_" + scale_radius + "_" + scale_height + "_" + pos_angle + ".fits"
            print("Comparing " + mode + " against " + image_filename + "...")

            # Get float values for the details
            flux = float(flux)
            inc_angle = float(inc_angle)
            scale_radius = float(scale_radius)
            scale_height = float(scale_height)
            pos_angle = float(pos_angle)

            image = galsim.fits.read(image_filename, image_dir)
            image *= flux / image.array.sum()
            nx, ny = np.shape(image.array)

            # Now make a test image
            test_profile = get_prof(mode, inc_angle * galsim.radians, scale_radius,
                                    scale_height, flux=flux)

            gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
            test2 = get_prof(mode, inc_angle * galsim.radians, scale_radius, scale_height,
                             flux=flux, gsparams=gsp)
            assert test2 != test_profile
            assert test2 == test_profile.withGSParams(gsp)
            assert test2 == test_profile.withGSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)

            check_basic(test_profile, mode)

            # Rotate it by the position angle
            test_profile = test_profile.rotate(pos_angle * galsim.radians)

            # Draw it onto an image
            test_image = galsim.Image(np.zeros_like(image.array), scale=1.0)
            test_profile.drawImage(test_image, offset=(0.5, 0.5)) # Offset to match Lance's

            # Save for manual inspection if desired
            if save_profiles:
                test_image_filename = image_filename.replace(".fits", "_" + mode + ".fits")
                test_image.write(test_image_filename, image_dir, clobber=True)

            # Compare to the example - Due to the different fourier transforms used, some offset is
            # expected, so we just compare in the core to two decimal places

            image_core = image.array[ nx//2-2 : nx//2+3 , ny//2-2 : ny//2+3 ]
            test_image_core = test_image.array[ nx//2-2 : nx//2+3 , ny//2-2 : ny//2+3 ]

            # Be a bit more lenient in the edge-on case, since it has greater errors in the FFT
            if np.cos(inc_angle) < 0.01:
                rtol = 5e-2
                atol = 5e-4
            else:
                rtol = 1e-2
                atol = 1e-4

            np.testing.assert_allclose(
                    test_image_core, image_core,
                    rtol=rtol, atol=atol * flux,
                    err_msg="Error in comparison of " + mode + " profile to " + image_filename,
                    verbose=True)

    # Now do Sersic-only tests
    for (flux, sersic_n, inc_angle, scale_radius, scale_height,
         trunc_factor, pos_angle) in inclined_sersic_regression_test_parameters:

        image_filename = ("galaxy_" + sersic_n + "_" + inc_angle + "_" + scale_radius +
                          "_" + scale_height + "_" + trunc_factor + "_" + pos_angle + ".fits")
        print("Comparing " + mode + " against " + image_filename + "...")

        # Get float values for the details
        flux = float(flux)
        sersic_n = float(sersic_n)
        inc_angle = float(inc_angle)
        scale_radius = float(scale_radius)
        scale_height = float(scale_height)
        trunc_factor = float(trunc_factor)
        pos_angle = float(pos_angle)

        image = galsim.fits.read(image_filename, image_dir)
        image *= flux / image.array.sum()
        nx, ny = np.shape(image.array)

        # Now make a test image
        test_profile = get_prof("InclinedSersic", n=sersic_n, scale_radius=scale_radius, scale_height=scale_height,
                                inclination=inc_angle * galsim.radians, trunc=trunc_factor * scale_radius, flux=flux)
        check_basic(test_profile, mode)

        # Rotate it by the position angle
        test_profile = test_profile.rotate(pos_angle * galsim.radians)

        # Draw it onto an image
        test_image = galsim.Image(nx, ny, scale=1.0)
        test_profile.drawImage(test_image, offset=(0.5, 0.5)) # Offset to match Lance's

        # Save if desired
        if save_profiles:
            test_image_filename = image_filename.replace(".fits", "_" + mode + ".fits")
            test_image.write(test_image_filename, image_dir, clobber=True)

        # Compare to the example - Due to the different fourier transforms used, some offset is
        # expected, so we just compare in the core to two decimal places

        image_core = image.array[nx // 2 - 2:nx // 2 + 3, ny // 2 - 2:ny // 2 + 3]
        test_image_core = test_image.array[nx // 2 - 2:nx // 2 + 3, ny // 2 - 2:ny // 2 + 3]

        np.testing.assert_allclose(
                test_image_core, image_core,
                rtol=1e-2, atol=1e-4 * flux,
                err_msg="Error in comparison of " + mode + " profile to " + image_filename,
                verbose=True)


@timer
def test_exponential():
    """ Test that InclinedExponential looks identical to an exponential when inclination is zero.
    """

    scale_radius = 3.0
    hlr = 1.7
    mode = "InclinedExponential"

    # Construct from scale_radius
    exp_profile = galsim.Exponential(scale_radius=scale_radius)
    inc_profile = get_prof(mode, 0 * galsim.radians, scale_radius=scale_radius,
                           scale_height=scale_radius / 10.)
    np.testing.assert_almost_equal(inc_profile.scale_radius, exp_profile.scale_radius)
    np.testing.assert_almost_equal(inc_profile.disk_half_light_radius,
                                   exp_profile.half_light_radius)

    # Construct from half_light_radius
    exp_profile = galsim.Exponential(half_light_radius=hlr)
    inc_profile = get_prof(mode, 0 * galsim.radians, half_light_radius=hlr,
                           scale_height=scale_radius / 10.)
    np.testing.assert_almost_equal(inc_profile.scale_radius, exp_profile.scale_radius)
    np.testing.assert_almost_equal(inc_profile.disk_half_light_radius,
                                   exp_profile.half_light_radius)

    exp_image = galsim.Image(image_nx, image_ny, scale=1.0)
    exp_profile.drawImage(exp_image)
    inc_image = galsim.Image(image_nx, image_ny, scale=1.0)
    inc_profile.drawImage(inc_image)

    # Check that they're the same
    np.testing.assert_array_almost_equal(inc_image.array, exp_image.array, decimal=4)

    # The face-on version should get the max_sb value exactly right
    np.testing.assert_array_almost_equal(inc_profile.max_sb, exp_profile.max_sb)

    check_basic(inc_profile, "Face-on " + mode)


@timer
def test_sersic():
    """ Test that InclinedSersic looks identical to a Sersic when inclination is zero.
    """

    ns = (1.1, 1.1, 2.5, 2.5)
    truncs = (0, 13.5, 0, 18.0)
    scale_radius = 3.0
    hlr = 1.7
    mode = "InclinedSersic"

    for n, trunc in zip(ns, truncs):

        # Construct from scale_radius
        sersic_profile = galsim.Sersic(n=n, scale_radius=scale_radius, trunc=trunc)
        inc_profile = get_prof(mode, n=n, trunc=trunc, inclination=0 * galsim.radians,
                               scale_radius=scale_radius,
                               scale_height=hlr / 10.)
        np.testing.assert_almost_equal(inc_profile.scale_radius, sersic_profile.scale_radius)
        np.testing.assert_almost_equal(inc_profile.disk_half_light_radius,
                                       sersic_profile.half_light_radius)

        # Construct from half-light radius
        sersic_profile = galsim.Sersic(n=n, half_light_radius=hlr, trunc=trunc)
        inc_profile = get_prof(mode, n=n, trunc=trunc, inclination=0 * galsim.radians,
                               half_light_radius=hlr,
                               scale_height=hlr / 10.)
        np.testing.assert_almost_equal(inc_profile.scale_radius, sersic_profile.scale_radius)
        np.testing.assert_almost_equal(inc_profile.disk_half_light_radius,
                                       sersic_profile.half_light_radius)

        sersic_image = galsim.Image(image_nx, image_ny, scale=1.0)
        sersic_profile.drawImage(sersic_image)
        inc_image = galsim.Image(image_nx, image_ny, scale=1.0)
        inc_profile.drawImage(inc_image)

        if save_profiles:
            sersic_image.write("test_sersic.fits", image_dir, clobber=True)
            inc_image.write("test_inclined_sersic.fits", image_dir, clobber=True)

        # Check that they're the same. Note that since the inclined Sersic profile isn't
        # Real-space analytic and has hard edges in the truncated case,
        # we have to be a bit lax on rtol and atol
        if trunc != 0:
            rtol = 5e-3
            atol = 5e-5
        else:
            rtol = 1e-3
            atol = 1e-5

        np.testing.assert_allclose(inc_image.array, sersic_image.array, rtol=rtol, atol=atol)

        # The face-on version should get the max_sb value exactly right
        np.testing.assert_almost_equal(inc_profile.max_sb, sersic_profile.max_sb)

        check_basic(inc_profile, "Face-on " + mode)


@timer
def test_edge_on():
    """ Test that an edge-on profile looks similar to an almost-edge-on profile, and doesn't crash.
    """
    from scipy.special import gamma

    scale_radius = 3.0
    sersic_n = 2.0

    inclinations = (np.arccos(0.01), 2 * np.pi - np.arccos(0.01), np.pi / 2.)

    for mode in ("InclinedExponential", "InclinedSersic"):

        if mode == "InclinedExponential":
            n = 1.0
            comp_prof = galsim.Exponential(scale_radius=scale_radius)
        else:
            n = sersic_n
            comp_prof = galsim.Sersic(n=n, scale_radius=scale_radius)

        images = []

        for inclination in inclinations:
            # Set up the profile
            prof = get_prof(mode, inclination * galsim.radians, scale_radius=scale_radius,
                            scale_h_over_r=0.1, n=n)

            check_basic(prof, "Edge-on " + mode)

            # Draw an image of it
            image = galsim.Image(image_nx, image_ny, scale=1.0)
            prof.drawImage(image)

            # Add it to the list of images
            images.append(image.array)

        # Check they're all almost the same
        np.testing.assert_array_almost_equal(images[1], images[0], decimal=2)
        np.testing.assert_array_almost_equal(images[1], images[2], decimal=2)

        # Also the edge-on version should get the max_sb value exactly right
        np.testing.assert_allclose(prof.max_sb, comp_prof.max_sb * 10. * n / gamma(n))
        prof.drawImage(image, method='sb', use_true_center=False)
        print('max pixel: ', image.array.max(), ' cf.', prof.max_sb)
        np.testing.assert_allclose(image.array.max(), prof.max_sb, rtol=0.01)


@timer
def test_sanity():
    """ Performs various sanity checks on a set of InclinedExponential and InclinedSersic profiles. """

    def run_sanity_checks(mode, flux, inc_angle, scale_radius, scale_height, pos_angle,
                          n=1., trunc=0.):
        # Get float values for the details
        print(flux, inc_angle, scale_radius, scale_height, pos_angle, n, trunc)

        # Now make a test image
        test_profile = get_prof(mode, inc_angle * galsim.radians, scale_radius,
                                scale_height, flux=flux, n=n, trunc=trunc)

        check_basic(test_profile, mode)

        # Check accessing construction args
        np.testing.assert_equal(test_profile.inclination, inc_angle * galsim.radians)
        np.testing.assert_equal(test_profile.scale_radius, scale_radius)
        np.testing.assert_equal(test_profile.scale_height, scale_height)
        np.testing.assert_equal(test_profile.flux, flux)

        # Check that h/r is properly given by the method and property for it
        np.testing.assert_allclose(test_profile.scale_height / test_profile.scale_radius,
                                       test_profile.scale_h_over_r, rtol=1e-4)

        # Rotate it by the position angle
        test_profile = test_profile.rotate(pos_angle * galsim.radians)

        # Check that the k value for (0,0) is the flux
        np.testing.assert_allclose(test_profile.kValue(kx=0., ky=0.), flux, rtol=1e-4)

        # Check that the drawn flux for a large image is indeed the flux
        test_image = galsim.Image(int(5 * n ** 2) * image_nx, int(5 * n ** 2) * image_ny, scale=1.0)
        test_profile.drawImage(test_image)
        test_flux = test_image.array.sum()
        # Be a bit more lenient here for sersic profiles than exponentials
        if n == 1:
            rtol = 1e-4
        else:
            rtol = 1e-2
        np.testing.assert_allclose(test_flux, flux, rtol=rtol)

        # Check that the centroid is (0,0)
        centroid = test_profile.centroid
        np.testing.assert_equal(centroid.x, 0.)
        np.testing.assert_equal(centroid.y, 0.)

        # Check max_sb - just ensure it's not under by more than the empirical limit for this
        # approximation
        test_profile.drawImage(test_image, use_true_center=False)
        print('max pixel: ', test_image.array.max(), ' cf.', test_profile.max_sb)

        np.testing.assert_array_less(test_image.array.max(), test_profile.max_sb*1.56)

    # Run tests applicable to both profiles
    for mode in ("InclinedExponential", "InclinedSersic"):

        print('flux, inc_angle, scale_radius, scale_height, pos_angle, n, trunc')
        for (flux, _sersic_n, inc_angle, scale_radius, scale_height,
             _trunc_factor, pos_angle) in inclined_exponential_test_parameters:

            flux = float(flux)
            inc_angle = float(inc_angle)
            scale_radius = float(scale_radius)
            scale_height = float(scale_height)
            pos_angle = float(pos_angle)
            run_sanity_checks(mode, flux, inc_angle, scale_radius, scale_height, pos_angle)

    # Run tests for InclinedSersic only
    for (flux, sersic_n, inc_angle, scale_radius, scale_height,
         trunc_factor, pos_angle) in (inclined_sersic_test_parameters +
                                      inclined_sersic_regression_test_parameters):
        flux = float(flux)
        inc_angle = float(inc_angle)
        scale_radius = float(scale_radius)
        scale_height = float(scale_height)
        pos_angle = float(pos_angle)
        n = float(sersic_n)
        trunc = float(trunc_factor) * float(scale_radius)

        run_sanity_checks("InclinedSersic", flux, inc_angle, scale_radius, scale_height, pos_angle,
                          n=n, trunc=trunc)

        # Run specific tests for InclinedSersic

        # Check that flux_untruncated behaves as expected
        prof1a = get_prof("InclinedSersic",n=n,flux=flux,inclination=inc_angle*galsim.radians,
                         scale_radius=scale_radius,scale_height=scale_height,trunc=trunc)
        prof1b = get_prof("InclinedSersic",n=n,flux=flux,inclination=inc_angle*galsim.radians,
                         scale_radius=scale_radius,scale_height=scale_height,trunc=trunc,
                         flux_untruncated=False)
        prof2 = get_prof("InclinedSersic",n=n,flux=flux,inclination=inc_angle*galsim.radians,
                         scale_radius=scale_radius,scale_height=scale_height,trunc=trunc,
                         flux_untruncated=True)

        np.testing.assert_almost_equal(prof1a.flux, prof1b.flux, 9)
        print(flux, trunc_factor, trunc)
        print("   ", prof1a.flux, prof1b.flux, prof2.flux)
        if trunc > 0:
            assert(prof1a.flux > prof2.flux)


@timer
def test_k_limits(run_slow):
    """ Check that the maxk and stepk give reasonable results for all profiles. """

    test_params = inclined_sersic_regression_test_parameters
    if run_slow:
        test_params += inclined_exponential_test_parameters
        test_params += inclined_sersic_test_parameters

    for mode in ("InclinedExponential", "InclinedSersic"):

        for (_, _, inc_angle, scale_radius, scale_height, _, _) in test_params:

            # Get float values for the details
            inc_angle = float(inc_angle)
            scale_radius = float(scale_radius)
            scale_height = float(scale_height)

            gsparams = galsim.GSParams()

            # Now make a test image
            test_profile = get_prof(mode, inc_angle * galsim.radians, scale_radius, scale_height)

            # Check that the k value at maxk is below maxk_threshold in both the x and y dimensions
            kx = test_profile.maxk
            ky = test_profile.maxk

            kx_value = test_profile.kValue(kx=kx, ky=0.)
            np.testing.assert_(np.abs(kx_value) < gsparams.maxk_threshold,
                               msg="kx_value is not below maxk_threshold: " + str(kx_value) + " >= "
                                + str(gsparams.maxk_threshold))

            ky_value = test_profile.kValue(kx=0., ky=ky)
            np.testing.assert_(np.abs(ky_value) < gsparams.maxk_threshold,
                               msg="ky_value is not below maxk_threshold: " + str(ky_value) + " >= "
                                + str(gsparams.maxk_threshold))

            # Check that less than folding_threshold fraction of light falls outside r = pi/stepk
            rmax = np.pi / test_profile.stepk
            pixel_scale = 0.1
            test_image = galsim.Image(int(2*rmax/pixel_scale), int(2*rmax/pixel_scale),
                                      scale=pixel_scale)
            test_profile.drawImage(test_image)

            # Get an array of indices within the limits
            image_shape = np.shape(test_image.array)
            x, y = np.indices(image_shape, dtype=float)

            image_center = test_image.center
            x -= image_center.x
            y -= image_center.y

            # Include all pixels that are at least partially within distance r of the centre
            r = pixel_scale * np.sqrt(np.square(x) + np.square(y))
            good = r < rmax + np.sqrt(2.)*pixel_scale

            # Get flux within the limits
            contained_flux = np.ravel(test_image.array)[np.ravel(good)].sum()

            # Check that we're not missing too much flux
            total_flux = np.sum(test_image.array)
            assert (total_flux-contained_flux)/total_flux <= gsparams.folding_threshold


@timer
def test_eq_ne():
    """ Check that equality/inequality works as expected."""

    gsp = galsim.GSParams(folding_threshold=1.1e-3)

    diff_gals = []

    for mode in ("InclinedExponential", "InclinedSersic"):

        # First test that some different initializations that should be equivalent:
        same_gals = [get_prof(mode, 0.1 * galsim.radians, 3.0),
                get_prof(mode, 0.1 * galsim.radians, 3.0, 0.3), # default h/r = 0.1
                get_prof(mode, 0.1 * galsim.radians, 3.0, scale_height=0.3),
                get_prof(mode, 0.1 * galsim.radians, 3.0, scale_h_over_r=0.1),
                get_prof(mode, 0.1 * galsim.radians, 3.0, flux=1.0), # default flux=1
                get_prof(mode, -0.1 * galsim.radians, 3.0), # negative i is equivalent
                get_prof(mode, (np.pi - 0.1) * galsim.radians, 3.0), # also pi-theta
                get_prof(mode, 18. / np.pi * galsim.degrees, 3.0),
                get_prof(mode, inclination=0.1 * galsim.radians, scale_radius=3.0,
                         scale_height=0.3, flux=1.0),
                get_prof(mode, flux=1.0, scale_radius=3.0,
                         scale_height=0.3, inclination=0.1 * galsim.radians),
                get_prof(mode, flux=1.0, half_light_radius=3.0 * galsim.Exponential._hlr_factor,
                         scale_height=0.3, inclination=0.1 * galsim.radians),
                get_prof(mode, flux=1.0, half_light_radius=3.0 * galsim.Exponential._hlr_factor,
                         scale_h_over_r=0.1, inclination=0.1 * galsim.radians)]

        for gal in same_gals[1:]:
            print(gal)
            gsobject_compare(gal, same_gals[0])

        # Set up list of galaxies we expect to all be different
        diff_gals += [get_prof(mode, 0.1 * galsim.radians, 3.0, 0.3),
                get_prof(mode, 0.1 * galsim.degrees, 3.0, 0.3),
                get_prof(mode, 0.1 * galsim.degrees, 3.0, scale_h_over_r=0.2),
                get_prof(mode, 0.1 * galsim.radians, 3.0, 3.0),
                get_prof(mode, 0.2 * galsim.radians, 3.0, 0.3),
                get_prof(mode, 0.1 * galsim.radians, 3.1, 0.3),
                get_prof(mode, 0.1 * galsim.radians, 3.1),
                get_prof(mode, 0.1 * galsim.radians, 3.0, 0.3, flux=0.5),
                get_prof(mode, 0.1 * galsim.radians, 3.0, 0.3, gsparams=gsp)]

    # Add some more Sersic profiles to the diff_gals list
    diff_gals += [get_prof("InclinedSersic", 0.1 * galsim.radians, 3.0, 0.3, n=1.1),
                  get_prof("InclinedSersic", 0.1 * galsim.radians, 3.0, 0.3, trunc=4.5),
                  galsim.InclinedSersic(n=1.0, inclination=0.1 * galsim.radians, half_light_radius=3.0,
                                        scale_height=0.3),
                  galsim.InclinedSersic(n=1.0, inclination=0.1 * galsim.radians, scale_radius=3.0,
                                        scale_h_over_r=0.3)]

    check_all_diff(diff_gals)


@timer
def test_pickle():
    """ Check that we can pickle it. """

    for mode in ("InclinedExponential", "InclinedSersic"):

        prof = get_prof(mode, trunc=4.5, inclination=0.1 * galsim.radians, scale_radius=3.0,
                                         scale_height=0.3)
        check_pickle(prof)

        check_pickle(get_prof(mode, trunc=4.5, inclination=0.1 * galsim.radians, scale_radius=3.0))
        check_pickle(get_prof(mode, trunc=4.5, inclination=0.1 * galsim.radians, scale_radius=3.0,
                                             scale_h_over_r=0.2))
        check_pickle(get_prof(mode, trunc=4.5, inclination=0.1 * galsim.radians, scale_radius=3.0,
                                             scale_height=0.3, flux=10.0))
        check_pickle(get_prof(mode, trunc=4.5, inclination=0.1 * galsim.radians, scale_radius=3.0,
                                             scale_height=0.3,
                                             gsparams=galsim.GSParams(folding_threshold=1.1e-3)))
        check_pickle(get_prof(mode, trunc=4.5, inclination=0.1 * galsim.radians, scale_radius=3.0,
                                             scale_height=0.3, flux=10.0,
                                             gsparams=galsim.GSParams(folding_threshold=1.1e-3)))


@timer
def test_exceptions():
    """ Tests to make sure that proper exceptions are thrown when expected. """

    for mode in ("InclinedExponential", "InclinedSersic"):

        # Need at least one radius specification
        with assert_raises(TypeError):
            get_prof(mode, inclination = 0.*galsim.degrees)

        # Can't have two radius specifications
        with assert_raises(TypeError):
            get_prof(mode, inclination = 0.*galsim.degrees,
                     scale_radius = 1., half_light_radius = 1.)

        # Radius specification must be > 0
        with assert_raises(ValueError):
            get_prof(mode, inclination = 0.*galsim.degrees, scale_radius = -1.)
        with assert_raises(ValueError):
            get_prof(mode, inclination = 0.*galsim.degrees, half_light_radius = -1.)

        # Can't have both height specifications
        with assert_raises(TypeError):
            get_prof(mode, inclination = 0.*galsim.degrees,
                     scale_radius = 1., scale_height = 0.2, scale_h_over_r = 0.1)

        # Radius specification must be > 0
        with assert_raises(ValueError):
            get_prof(mode, inclination = 0.*galsim.degrees, scale_radius = 1., scale_height = -0.2)
        with assert_raises(ValueError):
            get_prof(mode, inclination = 0.*galsim.degrees,
                     scale_radius = 1., scale_h_over_r = -0.1)

        # Enforce inclination is an angle type
        with assert_raises(TypeError):
            get_prof(mode, inclination = 0., scale_radius = 1.)

    # Can't have negative truncation for InclinedSersic
    with assert_raises(ValueError):
        get_prof("InclinedSersic", inclination = 0.*galsim.degrees,
                 scale_radius = 1., trunc = -4.5)

    # trunc can't be too small in InclinedSersic
    with assert_raises(ValueError):
        get_prof("InclinedSersic", inclination = 0.*galsim.degrees,
                 half_light_radius = 1., trunc = 1.4)


@timer
def test_value_retrieval():
    """ Tests to make sure that if a parameter is passed to a profile, we get back the same
        value from it. Only parameters not tested by the pickling are tested here.
    """

    for mode in ("InclinedExponential", "InclinedSersic"):

        half_light_radius = 1.6342
        scale_h_over_r = 0.2435

        prof = get_prof(mode, inclination=0.*galsim.degrees, half_light_radius=half_light_radius,
                        scale_h_over_r=scale_h_over_r)

        np.testing.assert_almost_equal(half_light_radius, prof.disk_half_light_radius, 9)
        np.testing.assert_almost_equal(scale_h_over_r, prof.scale_h_over_r, 9)


if __name__ == "__main__":
    runtests(__file__)
