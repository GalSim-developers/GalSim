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

import numpy as np
import os

import galsim
from galsim_test_helpers import *

imgdir = os.path.join(".", "Optics_comparison_images") # Directory containing the reference images.


testshape = (512, 512)  # shape of image arrays for all tests

decimal = 6     # Last decimal place used for checking equality of float arrays, see
                # np.testing.assert_array_almost_equal(), low since many are ImageF

decimal_dft = 3  # Last decimal place used for checking near equality of DFT product matrices to
                 # continuous-result derived check values... note this is not as stringent as
                 # decimal, because this is tough, because the DFT representation of a function is
                 # not precisely equivalent to its continuous counterpart.
           # See http://en.wikipedia.org/wiki/File:From_Continuous_To_Discrete_Fourier_Transform.gif

# The lines below control the behavior of the tests that involve making PSFs from pupil plane
# images.  The best tests involve very high resolution images, but these are slow.  So, when running
# test_optics.py directly, you will get the slow tests (~25 minutes for all of them).  When running
# `scons tests`, you will get faster, less stringent tests.

do_slow_tests = False
# do_slow_tests = True   # uncomment out for more rigorous testing
#                         Warning: some of them require a LOT of memory.

if do_slow_tests:
    pp_decimal = 5
    pp_file = 'sample_pupil_rolled_oversample.fits.gz'
    pp_oversampling = 4.
    pp_pad_factor = 4.
    # In this case, we test the entire images.
    pp_test_type = 'image'
else:
    pp_decimal = 4
    pp_file = 'sample_pupil_rolled.fits'
    pp_oversampling = 1.5
    pp_pad_factor = 1.5
    # In the less stringent tests, we may opt to test only the 2nd moments rather than the images
    # themselves.  This is because when the original tests were set up, I found low-level artifacts
    # that made it hard to get image-based tests to pass, yet the adaptive moments agreed quite
    # well.  Given that there were no such problems in the image-based tests with high-res inputs, I
    # believe these artifacts in the low-res tests should be ignored (by doing moments-based tests
    # only).
    pp_test_type = 'moments'


@timer
def test_OpticalPSF_flux(run_slow):
    """Compare an unaberrated OpticalPSF flux to unity.
    """
    lods = (1.e-8, 4., 9.e5) # lambda/D values: don't choose unity in case symmetry hides something
    nlook = 512         # Need a bit bigger image than below to get enough flux
    image = galsim.ImageF(nlook,nlook)
    for lod in lods:
        optics_test = galsim.OpticalPSF(lam_over_diam=lod)
        optics_array = optics_test.drawImage(scale=.25*lod, image=image, method='no_pixel').array
        np.testing.assert_almost_equal(optics_array.sum(), 1., 2,
                err_msg="Unaberrated Optical flux not quite unity.")

        if run_slow:
            optics_test = galsim.OpticalPSF(lam_over_diam=lod, flux=177)
            optics_im = optics_test.drawImage(scale=.25*lod, image=image, method='no_pixel')
            np.testing.assert_almost_equal(optics_im.array.sum(), 177., 2)
            check_basic(optics_test, "OpticalPSF, flux=177")

            optics_test = galsim.OpticalPSF(lam_over_diam=lod, flux=-17)
            optics_im = optics_test.drawImage(scale=.25*lod, image=image, method='no_pixel')
            np.testing.assert_almost_equal(optics_im.array.sum(), -17., 2)
            check_basic(optics_test, "OpticalPSF, flux=-17")

    check_pickle(optics_test, lambda x: x.drawImage(nx=20, ny=20, scale=1.7, method='no_pixel'))
    check_pickle(optics_test)
    check_pickle(optics_test._psf)
    check_pickle(optics_test._psf, lambda x: x.drawImage(nx=20, ny=20, scale=1.7, method='no_pixel'))
    check_basic(optics_test, "OpticalPSF")
    assert optics_test._psf._screen_list.r0_500_effective is None
    assert optics_test._screen == optics_test._psf.screen_list[0]

    interpolant_test = galsim.OpticalPSF(lam_over_diam=4., interpolant='linear')
    check_pickle(interpolant_test)

    scale_unit_test = galsim.OpticalPSF(lam_over_diam=4., scale_unit=galsim.arcmin)
    check_pickle(scale_unit_test)

    gsparams_test = optics_test.withGSParams(minimum_fft_size=64)
    check_pickle(gsparams_test)

    with assert_raises(galsim.GalSimValueError):
        galsim.OpticalPSF(lam_over_diam=lods[0], fft_sign=0)
    psf = galsim.OpticalPSF(lam_over_diam=lods[0])
    assert psf.fft_sign == '+'
    psf = galsim.OpticalPSF(lam_over_diam=lods[0], fft_sign='-')
    assert psf.fft_sign == '-'


@timer
def test_OpticalPSF_vs_Airy():
    """Compare the array view on an unaberrated OpticalPSF to that of an Airy.
    """
    lods = (4.e-7, 9., 16.4) # lambda/D values: don't choose unity in case symmetry hides something
    nlook = 100
    image = galsim.ImageF(nlook,nlook)
    for lod in lods:
        airy_test = galsim.Airy(lam_over_diam=lod, obscuration=0., flux=1.)
        #pad same as an Airy, natch!
        optics_test = galsim.OpticalPSF(lam_over_diam=lod, pad_factor=1, suppress_warning=True)
        airy_array = airy_test.drawImage(scale=.25*lod, image=image, method='no_pixel').array
        optics_array = optics_test.drawImage(scale=.25*lod, image=image, method='no_pixel').array
        np.testing.assert_array_almost_equal(optics_array, airy_array, decimal_dft,
                err_msg="Unaberrated Optical not quite equal to Airy")


@timer
def test_OpticalPSF_vs_Airy_with_obs():
    """Compare the array view on an unaberrated OpticalPSF with obscuration to that of an Airy.
    """
    lod = 7.5    # lambda/D value: don't choose unity in case symmetry hides something
    obses = (0.1, 0.3, 0.5) # central obscuration radius ratios
    nlook = 100          # size of array region at the centre of each image to compare
    image = galsim.ImageF(nlook,nlook)
    for obs in obses:
        airy_test = galsim.Airy(lam_over_diam=lod, obscuration=obs, flux=2.)
        optics_test = galsim.OpticalPSF(lam_over_diam=lod, pad_factor=1, obscuration=obs,
                                        suppress_warning=True, flux=2.)
        airy_array = airy_test.drawImage(scale=1.,image=image, method='no_pixel').array
        optics_array = optics_test.drawImage(scale=1.,image=image, method='no_pixel').array
        np.testing.assert_array_almost_equal(optics_array, airy_array, decimal_dft,
                err_msg="Unaberrated Optical with obscuration not quite equal to Airy")
    check_pickle(optics_test, lambda x: x.drawImage(nx=20, ny=20, scale=1.7, method='no_pixel'))
    check_pickle(optics_test)


@timer
def test_OpticalPSF_aberrations_struts():
    """Test the generation of optical aberrations and struts against a known result.
    """
    lod = 0.04
    obscuration = 0.3
    imsize = 128 # Size of saved images as generated by generate_optics_comparison_images.py
    myImg = galsim.ImageD(imsize, imsize)

    # We don't bother running all of these for the regular unit tests, since it adds
    # ~5s to the test run time on a fast-ish laptop.  So only run these when individually
    # running python test_optics.py.
    if do_slow_tests:
        # test defocus
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_defocus.fits"))
        optics = galsim.OpticalPSF(lod, defocus=.5, obscuration=obscuration, oversampling=1,
                                   fft_sign='-')
        myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')

        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (defocus) disagrees with expected result")

        # test astig1
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_astig1.fits"))
        optics = galsim.OpticalPSF(lod, defocus=.5, astig1=.5, obscuration=obscuration,
                                   oversampling=1, fft_sign='-')
        myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (astig1) disagrees with expected result")

        # test astig2
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_astig2.fits"))
        optics = galsim.OpticalPSF(lod, defocus=.5, astig2=.5, obscuration=obscuration,
                                   oversampling=1, fft_sign='-')
        myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (astig2) disagrees with expected result")

        # test coma1
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_coma1.fits"))
        optics = galsim.OpticalPSF(lod, coma1=.5, obscuration=obscuration, oversampling=1,
                                   fft_sign='-')
        myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (coma1) disagrees with expected result")

        # test coma2
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_coma2.fits"))
        optics = galsim.OpticalPSF(lod, coma2=.5, obscuration=obscuration, oversampling=1,
                                   fft_sign='-')
        myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (coma2) disagrees with expected result")

        # test trefoil1
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_trefoil1.fits"))
        optics = galsim.OpticalPSF(lod, trefoil1=.5, obscuration=obscuration, oversampling=1,
                                   fft_sign='-')
        myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (trefoil1) disagrees with expected result")

        # test trefoil2
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_trefoil2.fits"))
        optics = galsim.OpticalPSF(lod, trefoil2=.5, obscuration=obscuration, oversampling=1,
                                   fft_sign='-')
        myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (trefoil2) disagrees with expected result")

        # test spherical
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_spher.fits"))
        optics = galsim.OpticalPSF(lod, spher=.5, obscuration=obscuration, oversampling=1,
                                   fft_sign='-')
        myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (spher) disagrees with expected result")

    # test all aberrations
    savedImg = galsim.fits.read(os.path.join(imgdir, "optics_all.fits"))
    optics = galsim.OpticalPSF(lod, defocus=.5, astig1=0.5, astig2=0.3, coma1=0.4, coma2=-0.3,
                               trefoil1=-0.2, trefoil2=0.1, spher=-0.8, obscuration=obscuration,
                               oversampling=1, fft_sign='-')
    myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')
    np.testing.assert_array_almost_equal(
        myImg.array, savedImg.array, 6,
        err_msg="Optical aberration (all aberrations) disagrees with expected result")
    check_pickle(optics, lambda x: x.drawImage(nx=20, ny=20, scale=1.7, method='no_pixel'))
    check_pickle(optics)

    # test struts
    savedImg = galsim.fits.read(os.path.join(imgdir, "optics_struts.fits"))
    optics = galsim.OpticalPSF(
        lod, obscuration=obscuration, nstruts=5, strut_thick=0.04, strut_angle=8.*galsim.degrees,
        astig2=0.04, coma1=-0.07, defocus=0.09, oversampling=1, fft_sign='-')
    with assert_raises(TypeError):
        galsim.OpticalPSF(lod, nstruts=5, strut_thick=0.01, strut_angle=8.) # wrong units
    check_pickle(optics, lambda x: x.drawImage(nx=20, ny=20, scale=1.7, method='no_pixel'))
    check_pickle(optics)

    # Make sure it doesn't have some weird error if strut_angle=0 (should be the easiest case, but
    # check anyway...)
    optics_2 = galsim.OpticalPSF(
        lod, obscuration=obscuration, nstruts=4, strut_thick=0.05, strut_angle=0.*galsim.degrees,
        astig2=0.04, coma1=-0.07, defocus=0.09, oversampling=1, fft_sign='-')
    myImg = optics.drawImage(myImg, scale=0.2*lod, use_true_center=True, method='no_pixel')
    np.testing.assert_array_almost_equal(
        myImg.array, savedImg.array, 6,
        err_msg="Optical PSF (with struts) disagrees with expected result")
    # These are also the defaults for strut_thick and strut_angle
    optics_3 = galsim.OpticalPSF(
        lod, obscuration=obscuration, nstruts=4,
        astig2=0.04, coma1=-0.07, defocus=0.09, oversampling=1, fft_sign='-')
    assert optics_3 == optics_2
    check_pickle(optics_3)

    # make sure it doesn't completely explode when asked to return a PSF with non-circular pupil and
    # non-zero obscuration
    optics = galsim.OpticalPSF(
        lod, obscuration=obscuration, nstruts=5, strut_thick=0.04, strut_angle=8.*galsim.degrees,
        astig2=0.04, coma1=-0.07, defocus=0.09, circular_pupil=False, oversampling=1, fft_sign='-')
    check_pickle(optics, lambda x: x.drawImage(nx=20, ny=20, scale=1.7, method='no_pixel'))
    check_pickle(optics)


@timer
def test_OpticalPSF_aberrations_kwargs():
    """Test that OpticalPSF aberrations kwarg works just like specifying aberrations.
    """
    # Make an OpticalPSF with direct specification of aberrations.
    lod = 0.04
    obscuration = 0.3
    opt1 = galsim.OpticalPSF(lod, obscuration=obscuration, defocus=0.1, coma1=-0.1, coma2=0.3)

    # Now make it with an aberrations list.  (Note: should work with len < 12)
    aberrations = np.zeros(9)
    aberrations[4] = 0.1
    aberrations[7] = -0.1
    aberrations[8] = 0.3

    opt2 = galsim.OpticalPSF(lod, obscuration=obscuration, aberrations=aberrations)

    # Make sure they agree.
    np.testing.assert_array_equal(
        opt1.drawImage(scale=0.2*lod, method='no_pixel').array,
        opt2.drawImage(scale=0.2*lod, method='no_pixel').array,
        err_msg="Optical PSF depends on how aberrations are specified (4,8,11)")

    # Repeat with all aberrations up to index 11, using a regular list, not a numpy array
    opt1 = galsim.OpticalPSF(lod, defocus=.5, astig1=0.5, astig2=0.3, coma1=0.4, coma2=-0.3,
                             trefoil1=-0.2, trefoil2=0.1, spher=-0.8, obscuration=obscuration)
    aberrations = [ 0.0 ] * 4 + [ 0.5, 0.5, 0.3, 0.4, -0.3, -0.2, 0.1, -0.8 ]
    opt2 = galsim.OpticalPSF(lod, obscuration=obscuration, aberrations=aberrations)
    np.testing.assert_array_equal(
        opt1.drawImage(scale=0.2*lod, method='no_pixel').array,
        opt2.drawImage(scale=0.2*lod, method='no_pixel').array,
        err_msg="Optical PSF depends on how aberrations are specified (full list)")
    check_pickle(opt2, lambda x: x.drawImage(nx=20, ny=20, scale=0.01, method='no_pixel'))
    check_pickle(opt2)

    # Also, check for proper response to weird inputs.
    # aberrations must be a list or an array
    with assert_raises(TypeError):
        galsim.OpticalPSF(lod, aberrations=0.3)
    # It must have at least 2 elements
    with assert_raises(ValueError):
        galsim.OpticalPSF(lod, aberrations=[0.0])
    with assert_raises(ValueError):
        galsim.OpticalPSF(lod, aberrations=[])
    # 2 zeros is equivalent to None
    assert galsim.OpticalPSF(lod, aberrations=[0, 0]) == galsim.OpticalPSF(lod)
    # The first element must be 0. (Just a warning!)
    with assert_warns(galsim.GalSimWarning):
        galsim.OpticalPSF(lod, aberrations=[0.3]*8)
    # Cannot provide both aberrations and specific ones by name.
    with assert_raises(TypeError):
        galsim.OpticalPSF(lod, aberrations=np.zeros(8), defocus=-0.12)


@timer
def test_OpticalPSF_flux_scaling():
    """Test flux scaling for OpticalPSF.
    """
    # OpticalPSF test params (only a selection)
    test_flux = 1.8
    test_loD = 1.9
    test_obscuration = 0.32
    test_defocus = -0.7
    test_astig1 = 0.03
    test_astig2 = -0.04
    test_oversampling = 1.3
    test_pad_factor = 1.7

    # decimal point to go to for parameter value comparisons
    param_decimal = 12

    # init
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling,pad_factor=test_pad_factor,
        defocus=test_defocus, astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj *= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, pad_factor=test_pad_factor,
        defocus=test_defocus, astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, pad_factor=test_pad_factor,
        defocus=test_defocus, astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, pad_factor=test_pad_factor,
        defocus=test_defocus, astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, pad_factor=test_pad_factor,
        defocus=test_defocus, astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed...
    np.testing.assert_almost_equal(
        obj.flux, test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.flux, test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")


@timer
def test_OpticalPSF_pupil_plane():
    """Test the ability to generate a PSF using an image of the pupil plane.
    """
    # Test case: lam/diam=0.12, obscuration=0.18, 4 struts of the default width and with rotation
    # from the vertical of -15 degrees.  There are two versions of these tests at different
    # oversampling levels.
    #
    # To (re-)generate the pupil plane images for this test, simply delete
    # tests/Optics_comparison_images/sample_pupil_rolled.fits and
    # tests/Optics_comparison_images/sample_pupil_rolled_oversample.fits.gz,
    # and then rerun this function.  Note that these images are also used in test_ne(), so there
    # may be some racing if this script is tested in parallel before the fits files are regenerated.

    # First test: should get excellent agreement between that particular OpticalPSF with specified
    # options and one from loading the pupil plane image.  Note that this won't work if you change
    # the optical PSF parameters, unless you also regenerate the test image.
    lam_over_diam = 0.12
    obscuration = 0.18
    nstruts = 4
    strut_angle = -15.*galsim.degrees
    scale = 0.055
    ref_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, nstruts=nstruts,
                                oversampling=pp_oversampling, strut_angle=strut_angle,
                                pad_factor=pp_pad_factor)
    if os.path.isfile(os.path.join(imgdir, pp_file)):
        im = galsim.fits.read(os.path.join(imgdir, pp_file))
    else:
        import warnings
        warnings.warn("Could not find file {0}, so generating it from scratch.  This should only "
                      "happen if you intentionally deleted the file in order to regenerate it!"
                      .format(pp_file))
        im = galsim.Image(ref_psf._psf.aper.illuminated.astype(float))
        im.scale = ref_psf._psf.aper.pupil_plane_scale
        print('pupil_plane image has scale = ',im.scale)
        im.write(os.path.join(imgdir, pp_file))
    pp_scale = im.scale
    print('pupil_plane image has scale = ',pp_scale)

    # For most of the tests, we remove this scale, since for achromatic tests, you don't really
    # need it, and it is invalid to give lam_over_diam (rather than lam, diam separately) when
    # there is a specific scale for the pupil plane image.  But see the last test below where
    # we do use lam, diam separately with the input image.
    im.wcs = None
    # This implies that the lam_over_diam value is valid.
    test_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration,
                                 oversampling=pp_oversampling, pupil_plane_im=im,
                                 pad_factor=pp_pad_factor)
    im_ref_psf = ref_psf.drawImage(scale=scale)
    im_test_psf = galsim.ImageD(im_ref_psf.array.shape[0], im_ref_psf.array.shape[1])
    im_test_psf = test_psf.drawImage(image=im_test_psf, scale=scale)

    if pp_test_type == 'image':
        np.testing.assert_array_almost_equal(
            im_test_psf.array, im_ref_psf.array, decimal=pp_decimal,
            err_msg="Inconsistent OpticalPSF image for basic model after loading pupil plane.")
    else:
        test_moments = im_test_psf.FindAdaptiveMom()
        ref_moments = im_ref_psf.FindAdaptiveMom()
        np.testing.assert_almost_equal(
            test_moments.moments_sigma, ref_moments.moments_sigma, decimal=pp_decimal,
            err_msg="Inconsistent OpticalPSF image for basic model after loading pupil plane.")

    if do_slow_tests:
        check_pickle(test_psf, lambda x: x.drawImage(nx=20, ny=20, scale=0.07, method='no_pixel'))
        check_pickle(test_psf)

    # Make a smaller pupil plane image to test the pickling of this, even without slow tests.
    factor = 4 if not do_slow_tests else 16
    with assert_warns(galsim.GalSimWarning):
        alt_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration,
                                    oversampling=1., pupil_plane_im=im.bin(factor,factor),
                                    pad_factor=1.)
        check_pickle(alt_psf)

    assert_raises(ValueError, galsim.OpticalPSF, lam_over_diam, pupil_plane_im='pp_file')
    assert_raises(ValueError, galsim.OpticalPSF, lam_over_diam, pupil_plane_im=im,
                  pupil_plane_scale=pp_scale)
    assert_raises(ValueError, galsim.OpticalPSF, lam_over_diam,
                  pupil_plane_im=im.view(scale=pp_scale))
    # These aren't raised until the image is actually used
    with assert_raises(ValueError):
        # not square
        op = galsim.OpticalPSF(lam_over_diam, pupil_plane_im=galsim.Image(im.array[:-2,:]))
        op.drawImage()
    with assert_raises(ValueError):
        # not even sides
        op = galsim.OpticalPSF(lam_over_diam, pupil_plane_im=galsim.Image(im.array[:-1,:-1]))
        op.drawImage()

    # It is supposed to be able to figure this out even if we *don't* tell it the pad factor. So
    # make sure that it still works even if we don't tell it that value.
    test_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, pupil_plane_im=im,
                                 oversampling=pp_oversampling)
    im_test_psf = galsim.ImageD(im_ref_psf.array.shape[0], im_ref_psf.array.shape[1])
    im_test_psf = test_psf.drawImage(image=im_test_psf, scale=scale)

    if pp_test_type == 'image':
        np.testing.assert_array_almost_equal(
            im_test_psf.array, im_ref_psf.array, decimal=pp_decimal,
            err_msg="Inconsistent OpticalPSF image for basic model after loading pupil plane without "
            "specifying parameters.")
    else:
        test_moments = im_test_psf.FindAdaptiveMom()
        ref_moments = im_ref_psf.FindAdaptiveMom()
        np.testing.assert_almost_equal(
            test_moments.moments_sigma, ref_moments.moments_sigma, decimal=pp_decimal,
            err_msg="Inconsistent OpticalPSF image for basic model after loading pupil plane without "
            "specifying parameters.")

    # Next test (less trivial): Rotate the struts by +27 degrees, and check that agreement is
    # good. This is making sure that the linear interpolation that is done when rotating does not
    # result in significant loss of accuracy.
    rot_angle = 27.*galsim.degrees
    ref_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, nstruts=nstruts,
                                strut_angle=strut_angle+rot_angle, oversampling=pp_oversampling,
                                pad_factor=pp_pad_factor)
    test_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, pupil_plane_im=im,
                                 pupil_angle=rot_angle, oversampling=pp_oversampling,
                                 pad_factor=pp_pad_factor)
    im_ref_psf = ref_psf.drawImage(scale=scale)
    im_test_psf = galsim.ImageD(im_ref_psf.array.shape[0], im_ref_psf.array.shape[1])
    im_test_psf = test_psf.drawImage(image=im_test_psf, scale=scale)
    # We are slightly less stringent here since it should not be exact.
    if pp_test_type == 'image':
        np.testing.assert_array_almost_equal(
            im_test_psf.array, im_ref_psf.array, decimal=pp_decimal-1,
            err_msg="Inconsistent OpticalPSF image for rotated model after loading pupil plane.")
    else:
        test_moments = im_test_psf.FindAdaptiveMom()
        ref_moments = im_ref_psf.FindAdaptiveMom()
        np.testing.assert_almost_equal(
            test_moments.moments_sigma, ref_moments.moments_sigma, decimal=pp_decimal-1,
            err_msg="Inconsistent OpticalPSF image for rotated model after loading pupil plane.")

    # Now include aberrations.  Here we are testing the ability to figure out the pupil plane extent
    # and sampling appropriately.  Those get fed into the routine for making the aberrations.
    defocus = -0.03
    coma1 = 0.03
    spher = -0.02
    ref_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, nstruts=nstruts,
                                strut_angle=strut_angle, defocus=defocus, coma1=coma1, spher=spher,
                                oversampling=pp_oversampling, pad_factor=pp_pad_factor)
    test_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, pupil_plane_im=im,
                                 defocus=defocus, coma1=coma1, spher=spher,
                                 oversampling=pp_oversampling, pad_factor=pp_pad_factor)
    im_ref_psf = ref_psf.drawImage(scale=scale)
    im_test_psf = galsim.ImageD(im_ref_psf.array.shape[0], im_ref_psf.array.shape[1])
    im_test_psf = test_psf.drawImage(image=im_test_psf, scale=scale)
    if pp_test_type == 'image':
        np.testing.assert_array_almost_equal(
            im_test_psf.array, im_ref_psf.array, decimal=pp_decimal,
            err_msg="Inconsistent OpticalPSF image for aberrated model after loading pupil plane.")
    else:
        test_moments = im_test_psf.FindAdaptiveMom()
        ref_moments = im_ref_psf.FindAdaptiveMom()
        np.testing.assert_almost_equal(
            test_moments.moments_sigma, ref_moments.moments_sigma, decimal=pp_decimal,
            err_msg="Inconsistent OpticalPSF image for aberrated model after loading pupil plane.")

    # Test for preservation of symmetries: the result should be the same if the pupil plane is
    # rotated by integer multiples of 2pi/(nstruts).
    ref_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, nstruts=nstruts,
                                strut_angle=strut_angle, oversampling=pp_oversampling,
                                pad_factor=pp_pad_factor)
    im_ref_psf = ref_psf.drawImage(scale=scale)
    for ind in range(1,nstruts):
        rot_angle = ind*2.*np.pi/nstruts
        test_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, pupil_plane_im=im,
                                     pupil_angle=rot_angle*galsim.radians,
                                     oversampling=pp_oversampling, pad_factor=pp_pad_factor)
        im_test_psf = galsim.ImageD(im_ref_psf.array.shape[0], im_ref_psf.array.shape[1])
        im_test_psf = test_psf.drawImage(image=im_test_psf, scale=scale)
        if pp_test_type == 'image':
            np.testing.assert_array_almost_equal(
                im_test_psf.array, im_ref_psf.array, decimal=pp_decimal,
                err_msg="Inconsistent OpticalPSF image after rotating pupil plane by invariant "
                "angle.")
        else:
            test_moments = im_test_psf.FindAdaptiveMom()
            ref_moments = im_test_psf.FindAdaptiveMom()
            np.testing.assert_almost_equal(
                test_moments.moments_sigma, ref_moments.moments_sigma, decimal=pp_decimal,
                err_msg="Inconsistent OpticalPSF image after rotating pupil plane by invariant "
                "angle.")

    # Test that if we rotate pupil plane with no aberrations, that's equivalent to rotating the PSF
    # itself.  Use rotation angle of 90 degrees so numerical issues due to the interpolation should
    # be minimal.
    rot_angle = 90.*galsim.degrees
    psf_1 = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, pupil_plane_im=im,
                              oversampling=pp_oversampling, pad_factor=pp_pad_factor)
    rot_psf_1 = psf_1.rotate(rot_angle)
    psf_2 = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, pupil_plane_im=im,
                              pupil_angle=rot_angle, oversampling=pp_oversampling,
                              pad_factor=pp_pad_factor)
    im_1 = psf_1.drawImage(scale=scale)
    im_2 = galsim.ImageD(im_1.array.shape[0], im_1.array.shape[1])
    im_2 = psf_2.drawImage(image=im_2, scale=scale)
    if pp_test_type == 'image':
        np.testing.assert_array_almost_equal(
            im_1.array, im_2.array, decimal=pp_decimal,
            err_msg="Inconsistent OpticalPSF image after rotating pupil plane vs. rotating PSF.")
    else:
        test_moments = im_1.FindAdaptiveMom()
        ref_moments = im_2.FindAdaptiveMom()
        np.testing.assert_almost_equal(
            test_moments.moments_sigma, ref_moments.moments_sigma, decimal=pp_decimal,
            err_msg="Inconsistent OpticalPSF image after rotating pupil plane vs. rotating PSF.")

    # Supply the pupil plane at higher resolution, and make sure that the routine figures out the
    # sampling and gets the right image scale etc.
    rescale_fac = 0.77
    ref_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, nstruts=nstruts,
                                strut_angle=strut_angle, oversampling=pp_oversampling,
                                pad_factor=pp_pad_factor/rescale_fac)
    # Make higher resolution pupil plane image via interpolation
    int_im = galsim.InterpolatedImage(galsim.Image(im, scale=1.0, dtype=np.float32),
                                      calculate_maxk=False, calculate_stepk=False,
                                      x_interpolant='linear')
    new_im = int_im.drawImage(scale=rescale_fac, method='no_pixel')
    new_im.wcs = None  # Let OpticalPSF figure out the scale automatically.
    test_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration,
                                 pupil_plane_im=new_im, oversampling=pp_oversampling)
    im_ref_psf = ref_psf.drawImage(scale=scale)
    im_test_psf = galsim.ImageD(im_ref_psf.array.shape[0], im_ref_psf.array.shape[1])
    im_test_psf = test_psf.drawImage(image=im_test_psf, scale=scale)
    test_moments = im_test_psf.FindAdaptiveMom()
    ref_moments = im_ref_psf.FindAdaptiveMom()
    if pp_test_type == 'image':
        np.testing.assert_almost_equal(
            test_moments.moments_sigma/ref_moments.moments_sigma-1., 0, decimal=2,
            err_msg="Inconsistent OpticalPSF image for basic model after loading high-res pupil plane.")
    else:
        np.testing.assert_almost_equal(
            test_moments.moments_sigma/ref_moments.moments_sigma-1., 0, decimal=1,
            err_msg="Inconsistent OpticalPSF image for basic model after loading high-res pupil plane.")

    # Now supply the pupil plane at the original resolution, but remove some of the padding.  We
    # want it to properly recognize that it needs more padding, and include it.
    remove_pad = -23
    sub_im = im[im.bounds.withBorder(remove_pad)]
    test_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration,
                                 pupil_plane_im=sub_im, oversampling=pp_oversampling,
                                 pad_factor=pp_pad_factor)
    im_test_psf = galsim.ImageD(im_ref_psf.array.shape[0], im_ref_psf.array.shape[1])
    im_test_psf = test_psf.drawImage(image=im_test_psf, scale=scale)
    test_moments = im_test_psf.FindAdaptiveMom()
    ref_moments = im_ref_psf.FindAdaptiveMom()
    np.testing.assert_almost_equal(
        test_moments.moments_sigma/ref_moments.moments_sigma-1., 0, decimal=pp_decimal-3,
        err_msg="Inconsistent OpticalPSF image for basic model after loading less padded pupil plane.")

    # Now supply the pupil plane at the original resolution, with extra padding.
    new_pad = 76
    big_im = galsim.Image(im.bounds.withBorder(new_pad))
    big_im[im.bounds] = im
    test_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration,
                                 pupil_plane_im=big_im, oversampling=pp_oversampling,
                                 pad_factor=pp_pad_factor)
    im_test_psf = galsim.ImageD(im_ref_psf.array.shape[0], im_ref_psf.array.shape[1])
    im_test_psf = test_psf.drawImage(image=im_test_psf, scale=scale)
    test_moments = im_test_psf.FindAdaptiveMom()
    ref_moments = im_ref_psf.FindAdaptiveMom()
    np.testing.assert_almost_equal(
        test_moments.moments_sigma, ref_moments.moments_sigma, decimal=pp_decimal-2,
        err_msg="Inconsistent OpticalPSF image size for basic model "
        "after loading more padded pupil plane.")

    # Check for same answer if we use image, array, or filename for reading in array.
    test_psf = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, pupil_plane_im=im,
                                 oversampling=pp_oversampling, pad_factor=pp_pad_factor)
    im_test_psf = test_psf.drawImage(scale=scale)
    test_psf_2 = galsim.OpticalPSF(lam_over_diam, obscuration=obscuration, pupil_plane_im=im.array,
                                   oversampling=pp_oversampling, pad_factor=pp_pad_factor)
    im_test_psf_2 = test_psf_2.drawImage(scale=scale)
    np.testing.assert_almost_equal(
        im_test_psf.array, im_test_psf_2.array, decimal=pp_decimal,
        err_msg="Inconsistent OpticalPSF image from Image vs. array.")

    # The following had used lam_over_diam, but that is now invalid because the fits file
    # has a specific pixel scale.  So we need to provide lam and diam separately so that the
    # units are consistent.
    diam = 500.e-9 / lam_over_diam * galsim.radians / galsim.arcsec
    test_psf_3 = galsim.OpticalPSF(
        lam=500, diam=diam, obscuration=obscuration, oversampling=pp_oversampling,
        pupil_plane_im=os.path.join(imgdir, pp_file),
        pad_factor=pp_pad_factor)
    im_test_psf_3 = test_psf_3.drawImage(scale=scale)
    np.testing.assert_almost_equal(
        im_test_psf.array, im_test_psf_3.array, decimal=pp_decimal,
        err_msg="Inconsistent OpticalPSF image from Image vs. file read-in.")


@timer
def test_OpticalPSF_lamdiam():
    """Test the ability to generate an OpticalPSF using different lam/diam specifications.
    """
    # Choose some lam, diam, scale.
    lam = 457.3 # nm
    diam = 3.7 # m
    scale = 0.02*galsim.arcsec
    obscuration = 0.15

    # Make optical PSF using lam/diam and scale in some arbitrary units.  Let's use arcmin.
    lam_over_diam = 1.e-9*lam/diam*galsim.radians
    lam_over_diam_arcmin = lam_over_diam / galsim.arcmin
    opt_psf_1 = galsim.OpticalPSF(lam_over_diam = lam_over_diam_arcmin, obscuration=obscuration,
                                  scale_unit=galsim.arcmin)
    im_1 = opt_psf_1.drawImage(scale=scale/galsim.arcmin)

    # Make optical PSF using lam, diam separately and scale in arcsec.
    opt_psf_2 = galsim.OpticalPSF(lam=lam, diam=diam, obscuration=obscuration)
    im_2 = opt_psf_2.drawImage(scale=scale/galsim.arcsec)

    # These images should agree, since we defined PSF AND image scale using arcmin in one case and
    # using arcsec in the other case.
    np.testing.assert_almost_equal(
        im_2.array, im_1.array, decimal=8,
        err_msg="Inconsistent OpticalPSF when using different initialization arguments.")

    # Now make sure we cannot do some weird mix-and-match of arguments.
    assert_raises(TypeError, galsim.OpticalPSF, lam=1.) # need diam too!
    assert_raises(TypeError, galsim.OpticalPSF, diam=1.) # need lam too!
    assert_raises(TypeError, galsim.OpticalPSF, lam_over_diam=1., diam=1.)
    assert_raises(TypeError, galsim.OpticalPSF, lam_over_diam=1., lam=1.)


@timer
def test_OpticalPSF_pupil_plane_size():
    """Reproduce Chris Davis's test failure in (#752), but using a smaller, faster array."""
    im = galsim.Image(512, 512)
    x = y = np.arange(512) - 256
    y, x = np.meshgrid(y, x)
    im.array[x**2+y**2 < 230**2] = 1.0
    # The following still fails (uses deprecated optics framework):
    # galsim.optics.OpticalPSF(aberrations=[0,0,0,0,0.5], diam=4.0, lam=700.0, pupil_plane_im=im)
    # But using the new framework, should work.
    galsim.OpticalPSF(aberrations=[0,0,0,0,0.5], diam=4.0, lam=700.0, pupil_plane_im=im)


@timer
def test_OpticalPSF_aper():
    # Test setting up an OpticalPSF using an Aperture object instead of relying on the constructor
    # to initialize the aperture.
    lam = 500
    diam = 4.0

    aper = galsim.Aperture(lam=lam, diam=diam)
    psf1 = galsim.OpticalPSF(lam=lam, diam=diam, aper=aper)
    psf2 = galsim.OpticalPSF(lam=lam, diam=diam, oversampling=1.0, pad_factor=1.0)
    assert psf1 == psf2

    im = galsim.Image((psf1._aper.illuminated).astype(int))

    aper = galsim.Aperture(lam=lam, diam=diam, pupil_plane_im=im)
    psf1 = galsim.OpticalPSF(lam=lam, diam=diam, aper=aper)
    psf2 = galsim.OpticalPSF(lam=lam, diam=diam, pupil_plane_im=im,
                             oversampling=1.0, pad_factor=1.0)

    assert psf1 == psf2


@timer
def test_stepk_maxk_iipad():
    """Test options to specify (or not) stepk, maxk, and ii_pad_factor.
    """
    import time
    lam = 500
    diam = 4.0

    t0 = time.time()
    psf = galsim.OpticalPSF(lam=lam, diam=diam)
    print("Time for OpticalPSF with default ii_pad_factor=4 {0:6.4f}".format(time.time()-t0))
    stepk = psf.stepk
    maxk = psf.maxk

    psf2 = galsim.OpticalPSF(lam=lam, diam=diam, _force_stepk=stepk/1.5, _force_maxk=maxk*2.0)
    np.testing.assert_almost_equal(
            psf2.stepk, stepk/1.5, decimal=7,
            err_msg="OpticalPSF did not adopt forced value for stepk")
    np.testing.assert_almost_equal(
            psf2.maxk, maxk*2.0, decimal=7,
            err_msg="OpticalPSF did not adopt forced value for maxk")

    check_pickle(psf2)

    t0 = time.time()
    psf3 = galsim.OpticalPSF(lam=lam, diam=diam, ii_pad_factor=1.)
    print("Time for OpticalPSF with ii_pad_factor=1 {0:6.4f}".format(time.time()-t0))
    check_pickle(psf3)

    # The two images should be close, but not equivalent.
    im = psf.drawImage(nx=16, ny=16, scale=0.2)
    im3 = psf3.drawImage(nx=16, ny=16, scale=0.2)
    assert im != im3, (
            "Images drawn from InterpolatedImages with different pad_factor unexpectedly equal.")

    # Peak is ~0.2, to 1e-5 is pretty good.
    np.testing.assert_allclose(im.array, im3.array, rtol=0, atol=1e-5)


@timer
def test_ne():
    # Use some very forgiving settings to speed up this test.  We're not actually going to draw
    # any images (other than internally the PSF), so should be okay.
    gsp1 = galsim.GSParams(maxk_threshold=5.e-2, folding_threshold=5e-2, kvalue_accuracy=1e-3,
                           xvalue_accuracy=1e-3)
    gsp2 = galsim.GSParams(maxk_threshold=5.1e-2, folding_threshold=5e-2, kvalue_accuracy=1e-3,
                           xvalue_accuracy=1e-3)
    pupil_plane_im = galsim.fits.read(os.path.join(imgdir, pp_file))
    pupil_plane_im.wcs = None

    # Params include: lam_over_diam, (lam/diam), aberrations by name, aberrations by list, nstruts,
    # strut_thick, strut_angle, obscuration, oversampling, pad_factor, flux, gsparams,
    # circular_pupil, interpolant, pupil_plane_im, pupil_angle, scale_unit
    objs = [galsim.OpticalPSF(lam_over_diam=1.0, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, gsparams=gsp2),
            galsim.OpticalPSF(lam=1.0, diam=1.0, gsparams=gsp1),
            galsim.OpticalPSF(lam=1.0, diam=1.0, gsparams=gsp1, fft_sign='-'),
            galsim.OpticalPSF(lam=1.0, diam=1.0, scale_unit=galsim.arcmin, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, defocus=0.1, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, aberrations=[0, 0, 0, 0, 0.2], gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, nstruts=2, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, nstruts=2, strut_thick=0.3, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, nstruts=2, strut_angle=10.*galsim.degrees,
                              gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, obscuration=0.5, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, obscuration=0.5, coma1=1.0, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, obscuration=0.5, coma1=1.0, annular_zernike=True,
                              gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, oversampling=2.0, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, pad_factor=2.0, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, flux=2.0, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, circular_pupil=False, gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, interpolant='Linear', gsparams=gsp1),
            galsim.OpticalPSF(lam_over_diam=1.0, gsparams=gsp1, ii_pad_factor=2.)]
    stepk = objs[0].stepk
    maxk = objs[0].maxk
    objs += [galsim.OpticalPSF(lam_over_diam=1.0, gsparams=gsp1, _force_stepk=stepk/1.5),
             galsim.OpticalPSF(lam_over_diam=1.0, gsparams=gsp1, _force_maxk=maxk*2)]

    if do_slow_tests:
        objs += [galsim.OpticalPSF(lam_over_diam=1.0, pupil_plane_im=pupil_plane_im, gsparams=gsp1,
                                   suppress_warning=True),
                 galsim.OpticalPSF(lam_over_diam=1.0, pupil_plane_im=pupil_plane_im, gsparams=gsp1,
                                   pupil_angle=10*galsim.degrees, suppress_warning=True)]
    check_all_diff(objs)


@timer
def test_geometric_shoot():
    """Test that geometric photon shooting is reasonably consistent with Fourier optics."""
    jmax = 20
    bd = galsim.BaseDeviate(1111111)
    u = galsim.UniformDeviate(bd)

    lam = 500.0
    diam = 4.0

    for i in range(4):  # Do a few random tests.  Takes about 1 sec.
        aberrations = [0]+[u()*0.1 for i in range(jmax)]
        opt_psf = galsim.OpticalPSF(diam=diam, lam=lam, aberrations=aberrations,
                                    geometric_shooting=True)
        # Use really good seeing, so that the optics contribution actually matters.
        atm_psf = galsim.Kolmogorov(fwhm=0.3)

        psf = galsim.Convolve(opt_psf, atm_psf)
        u1 = u.duplicate()
        im_shoot = psf.drawImage(nx=256, ny=256, scale=0.2, method='phot', n_photons=100000, rng=u)
        im_fft = psf.drawImage(nx=256, ny=256, scale=0.2)

        printval(im_fft, im_shoot)
        shoot_moments = galsim.hsm.FindAdaptiveMom(im_shoot)
        fft_moments = galsim.hsm.FindAdaptiveMom(im_fft)

        # 40th of a pixel centroid tolerance.
        np.testing.assert_allclose(
            shoot_moments.moments_centroid.x, fft_moments.moments_centroid.x, rtol=0, atol=0.025,
            err_msg="")
        np.testing.assert_allclose(
            shoot_moments.moments_centroid.y, fft_moments.moments_centroid.y, rtol=0, atol=0.025,
            err_msg="")
        # 2% size tolerance
        np.testing.assert_allclose(
            shoot_moments.moments_sigma, fft_moments.moments_sigma, rtol=0.02, atol=0,
            err_msg="")
        # Not amazing ellipticity consistency at the moment.  0.01 tolerance.
        print(fft_moments.observed_shape)
        print(shoot_moments.observed_shape)
        np.testing.assert_allclose(
            shoot_moments.observed_shape.g1, fft_moments.observed_shape.g1, rtol=0, atol=0.01,
            err_msg="")
        np.testing.assert_allclose(
            shoot_moments.observed_shape.g2, fft_moments.observed_shape.g2, rtol=0, atol=0.01,
            err_msg="")

        # Check the flux
        # The Airy part sends a lot of flux off the edge, so this test is a little loose.
        added_flux = im_shoot.added_flux
        print('psf.flux = ',psf.flux)
        print('added_flux = ',added_flux)
        print('image flux = ',im_shoot.array.sum())
        assert np.isclose(added_flux, psf.flux, rtol=3.e-4)
        assert np.isclose(im_shoot.array.sum(), psf.flux, rtol=4.e-4)

        # Check doing this with photon_ops
        im_shoot2 = opt_psf.drawImage(nx=256, ny=256, scale=0.2, method='phot',
                                      n_photons=100000, rng=u1.duplicate(),
                                      photon_ops=[atm_psf])
        np.testing.assert_allclose(im_shoot2.array, im_shoot.array)
        im_shoot3 = galsim.DeltaFunction().drawImage(nx=256, ny=256, scale=0.2, method='phot',
                                                     n_photons=100000, rng=u1.duplicate(),
                                                     photon_ops=[opt_psf, atm_psf])
        np.testing.assert_allclose(im_shoot3.array, im_shoot.array)


if __name__ == "__main__":
    runtests(__file__)
