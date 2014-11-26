# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
import sys

from galsim_test_helpers import *

imgdir = os.path.join(".", "Optics_comparison_images") # Directory containing the reference images. 

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim


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
# test_optics.py directly, you will get the slow tests (~5 minutes for all of them).  When running
# `scons tests`, you will get faster, less stringent tests.
if __name__ == "__main__":
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

def test_check_all_contiguous():
    """Test all galsim.optics outputs are C-contiguous as required by the galsim.Image class.
    """
    import time
    t1 = time.time()
    # Check basic outputs from wavefront, psf and mtf (array contents won't matter, so we'll use
    # a pure circular pupil)
    test_obj, _ = galsim.optics.wavefront(array_shape=testshape)
    assert test_obj.flags.c_contiguous
    test_obj, _ = galsim.optics.psf(array_shape=testshape)
    assert test_obj.flags.c_contiguous
    assert galsim.optics.otf(array_shape=testshape).flags.c_contiguous
    assert galsim.optics.mtf(array_shape=testshape).flags.c_contiguous
    assert galsim.optics.ptf(array_shape=testshape).flags.c_contiguous
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_simple_wavefront():
    """Test the wavefront of a pure circular pupil against the known result.
    """
    import time
    t1 = time.time()
    kx, ky = galsim.utilities.kxky(testshape)
    dx_test = 3.  # } choose some properly-sampled, yet non-unit / trival, input params
    lod_test = 8. # }
    kmax_test = 2. * np.pi * dx_test / lod_test  # corresponding INTERNAL kmax used in optics code 
    kmag = np.sqrt(kx**2 + ky**2) / kmax_test # Set up array of |k| in units of kmax_test
    # Simple pupil wavefront should merely be unit ordinate tophat of radius kmax / 2: 
    in_pupil = kmag < .5
    wf_true = np.zeros(kmag.shape)
    wf_true[in_pupil] = 1.
    # Compare
    wf, _ = galsim.optics.wavefront(array_shape=testshape, scale=dx_test, lam_over_diam=lod_test)
    np.testing.assert_array_almost_equal(wf, wf_true, decimal=decimal)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_simple_mtf():
    """Test the MTF of a pure circular pupil against the known result.
    """
    import time
    t1 = time.time()
    kx, ky = galsim.utilities.kxky(testshape)
    dx_test = 3.  # } choose some properly-sampled, yet non-unit / trival, input params
    lod_test = 8. # }
    kmax_test = 2. * np.pi * dx_test / lod_test  # corresponding INTERNAL kmax used in optics code 
    kmag = np.sqrt(kx**2 + ky**2) / kmax_test # Set up array of |k| in units of kmax_test
    in_pupil = kmag < 1.
    # Then use analytic formula for MTF of circ pupil (fun to derive)
    mtf_true = np.zeros(kmag.shape)
    mtf_true[in_pupil] = (np.arccos(kmag[in_pupil]) - kmag[in_pupil] *
                          np.sqrt(1. - kmag[in_pupil]**2)) * 2. / np.pi
    # Compare
    mtf = galsim.optics.mtf(array_shape=testshape, scale=dx_test, lam_over_diam=lod_test)
    np.testing.assert_array_almost_equal(mtf, mtf_true, decimal=decimal_dft)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_simple_ptf():
    """Test the PTF of a pure circular pupil against the known result (zero).
    """
    import time
    t1 = time.time()
    ptf_true = np.zeros(testshape)
    # Compare
    ptf = galsim.optics.ptf(array_shape=testshape)
    # Test via median absolute deviation, since occasionally things around the edge of the OTF get
    # hairy when dividing a small number by another small number
    nmad_ptfdiff = np.median(np.abs(ptf - np.median(ptf_true)))
    assert nmad_ptfdiff <= 10.**(-decimal)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_consistency_psf_mtf():
    """Test that the MTF of a pure circular pupil is |FT{PSF}|.
    """
    import time
    t1 = time.time()
    kx, ky = galsim.utilities.kxky(testshape)
    dx_test = 3.  # } choose some properly-sampled, yet non-unit / trival, input params
    lod_test = 8. # }
    kmax_test = 2. * np.pi * dx_test / lod_test  # corresponding INTERNAL kmax used in optics code 
    psf, _ = galsim.optics.psf(array_shape=testshape, scale=dx_test, lam_over_diam=lod_test)
    psf *= dx_test**2 # put the PSF into flux units rather than SB for comparison
    mtf_test = np.abs(np.fft.fft2(psf))
    # Compare
    mtf = galsim.optics.mtf(array_shape=testshape, scale=dx_test, lam_over_diam=lod_test)
    np.testing.assert_array_almost_equal(mtf, mtf_test, decimal=decimal_dft)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_wavefront_image_view():
    """Test that the ImageF.array view of the wavefront is consistent with the wavefront array.
    """
    import time
    t1 = time.time()
    array, _ = galsim.optics.wavefront(array_shape=testshape)
    (real, imag), _ = galsim.optics.wavefront_image(array_shape=testshape)
    np.testing.assert_array_almost_equal(array.real.astype(np.float32), real.array, decimal)
    np.testing.assert_array_almost_equal(array.imag.astype(np.float32), imag.array, decimal)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_psf_image_view():
    """Test that the ImageF.array view of the PSF is consistent with the PSF array.
    """
    import time
    t1 = time.time()
    array, _ = galsim.optics.psf(array_shape=testshape)
    image = galsim.optics.psf_image(array_shape=testshape)
    np.testing.assert_array_almost_equal(array.astype(np.float32), image.array, decimal)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_otf_image_view():
    """Test that the ImageF.array view of the OTF is consistent with the OTF array.
    """
    import time
    t1 = time.time()
    array = galsim.optics.otf(array_shape=testshape)
    (real, imag) = galsim.optics.otf_image(array_shape=testshape)
    np.testing.assert_array_almost_equal(array.real.astype(np.float32), real.array, decimal)
    np.testing.assert_array_almost_equal(array.imag.astype(np.float32), imag.array, decimal)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_mtf_image_view():
    """Test that the ImageF.array view of the MTF is consistent with the MTF array.
    """
    import time
    t1 = time.time()
    array = galsim.optics.mtf(array_shape=testshape)
    image = galsim.optics.mtf_image(array_shape=testshape)
    np.testing.assert_array_almost_equal(array.astype(np.float32), image.array)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_ptf_image_view():
    """Test that the ImageF.array view of the OTF is consistent with the OTF array.
    """
    import time
    t1 = time.time()
    array = galsim.optics.ptf(array_shape=testshape)
    image = galsim.optics.ptf_image(array_shape=testshape)
    np.testing.assert_array_almost_equal(array.astype(np.float32), image.array)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_OpticalPSF_flux():
    """Compare an unaberrated OpticalPSF flux to unity.
    """
    import time
    t1 = time.time()
    lods = (1.e-8, 4., 9.e5) # lambda/D values: don't choose unity in case symmetry hides something
    nlook = 512         # Need a bit bigger image than below to get enough flux
    image = galsim.ImageF(nlook,nlook)
    for lod in lods:
        optics_test = galsim.OpticalPSF(lam_over_diam=lod)
        optics_array = optics_test.draw(scale=.25*lod, image=image).array 
        np.testing.assert_almost_equal(optics_array.sum(), 1., 2, 
                err_msg="Unaberrated Optical flux not quite unity.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_OpticalPSF_vs_Airy():
    """Compare the array view on an unaberrated OpticalPSF to that of an Airy.
    """
    import time
    t1 = time.time()
    lods = (4.e-7, 9., 16.4) # lambda/D values: don't choose unity in case symmetry hides something
    nlook = 100
    image = galsim.ImageF(nlook,nlook)
    for lod in lods:
        airy_test = galsim.Airy(lam_over_diam=lod, obscuration=0., flux=1.)
        #pad same as an Airy, natch!
        optics_test = galsim.OpticalPSF(lam_over_diam=lod, pad_factor=1, suppress_warning=True)
        airy_array = airy_test.draw(scale=.25*lod, image=image).array
        optics_array = optics_test.draw(scale=.25*lod, image=image).array 
        np.testing.assert_array_almost_equal(optics_array, airy_array, decimal_dft, 
                err_msg="Unaberrated Optical not quite equal to Airy")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_OpticalPSF_vs_Airy_with_obs():
    """Compare the array view on an unaberrated OpticalPSF with obscuration to that of an Airy.
    """
    import time
    t1 = time.time()
    lod = 7.5    # lambda/D value: don't choose unity in case symmetry hides something
    obses = (0.1, 0.3, 0.5) # central obscuration radius ratios
    nlook = 100          # size of array region at the centre of each image to compare
    image = galsim.ImageF(nlook,nlook)
    for obs in obses:
        airy_test = galsim.Airy(lam_over_diam=lod, obscuration=obs, flux=1.)
        optics_test = galsim.OpticalPSF(lam_over_diam=lod, pad_factor=1, obscuration=obs,
                                        suppress_warning=True)
        airy_array = airy_test.draw(scale=1.,image=image).array
        optics_array = optics_test.draw(scale=1.,image=image).array 
        np.testing.assert_array_almost_equal(optics_array, airy_array, decimal_dft, 
                err_msg="Unaberrated Optical with obscuration not quite equal to Airy")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_OpticalPSF_aberrations_struts():
    """Test the generation of optical aberrations and struts against a known result.
    """
    import time
    t1 = time.time()
    lod = 0.04
    obscuration = 0.3
    imsize = 128 # Size of saved images as generated by generate_optics_comparison_images.py
    myImg = galsim.ImageD(imsize, imsize)

    # We don't bother running all of these for the regular unit tests, since it adds
    # ~10s to the test run time on a fast-ish laptop.  So only run these when individually
    # running python test_optics.py.
    # NB: The test images were made with oversampling=1, so use that for these tests.
    if __name__ == "__main__":
        # test defocus
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_defocus.fits"))
        optics = galsim.OpticalPSF(lod, defocus=.5, obscuration=obscuration, oversampling=1)
        myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (defocus) disagrees with expected result")

        # test astig1
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_astig1.fits"))
        optics = galsim.OpticalPSF(lod, defocus=.5, astig1=.5, obscuration=obscuration, 
                                   oversampling=1)
        myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (astig1) disagrees with expected result")

        # test astig2
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_astig2.fits"))
        optics = galsim.OpticalPSF(lod, defocus=.5, astig2=.5, obscuration=obscuration, 
                                   oversampling=1)
        myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (astig2) disagrees with expected result")

        # test coma1
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_coma1.fits"))
        optics = galsim.OpticalPSF(lod, coma1=.5, obscuration=obscuration, oversampling=1)
        myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (coma1) disagrees with expected result")

        # test coma2
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_coma2.fits"))
        optics = galsim.OpticalPSF(lod, coma2=.5, obscuration=obscuration, oversampling=1)
        myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (coma2) disagrees with expected result")

        # test trefoil1
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_trefoil1.fits"))
        optics = galsim.OpticalPSF(lod, trefoil1=.5, obscuration=obscuration, oversampling=1)
        myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (trefoil1) disagrees with expected result")

        # test trefoil2
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_trefoil2.fits"))
        optics = galsim.OpticalPSF(lod, trefoil2=.5, obscuration=obscuration, oversampling=1)
        myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (trefoil2) disagrees with expected result")

        # test spherical
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_spher.fits"))
        optics = galsim.OpticalPSF(lod, spher=.5, obscuration=obscuration, oversampling=1)
        myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (spher) disagrees with expected result")

    # test all aberrations
    savedImg = galsim.fits.read(os.path.join(imgdir, "optics_all.fits"))
    optics = galsim.OpticalPSF(lod, defocus=.5, astig1=0.5, astig2=0.3, coma1=0.4, coma2=-0.3,
                               trefoil1=-0.2, trefoil2=0.1, spher=-0.8, obscuration=obscuration, 
                               oversampling=1)
    myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
    np.testing.assert_array_almost_equal(
        myImg.array, savedImg.array, 6,
        err_msg="Optical aberration (all aberrations) disagrees with expected result")

    # test struts
    savedImg = galsim.fits.read(os.path.join(imgdir, "optics_struts.fits"))
    optics = galsim.OpticalPSF(
        lod, obscuration=obscuration, nstruts=5, strut_thick=0.04, strut_angle=8.*galsim.degrees,
        astig2=0.04, coma1=-0.07, defocus=0.09, oversampling=1)
    try:
        np.testing.assert_raises(TypeError, galsim.OpticalPSF, lod, nstruts=5, strut_thick=0.01,
                                 strut_angle=8.) # wrong units
    except ImportError:
        print 'The assert_raises tests require nose'
    # Make sure it doesn't have some weird error if strut_angle=0 (should be the easiest case, but
    # check anyway...)
    optics_2 = galsim.OpticalPSF(
        lod, obscuration=obscuration, nstruts=5, strut_thick=0.04, strut_angle=0.*galsim.degrees,
        astig2=0.04, coma1=-0.07, defocus=0.09, oversampling=1)
    myImg = optics.draw(myImg, scale=0.2*lod, use_true_center=True)
    np.testing.assert_array_almost_equal(
        myImg.array, savedImg.array, 6,
        err_msg="Optical PSF (with struts) disagrees with expected result")

    # make sure it doesn't completely explode when asked to return a PSF with non-circular pupil and
    # non-zero obscuration
    optics = galsim.OpticalPSF(
        lod, obscuration=obscuration, nstruts=5, strut_thick=0.04, strut_angle=8.*galsim.degrees,
        astig2=0.04, coma1=-0.07, defocus=0.09, oversampling=1, circular_pupil=False)

    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2 - t1)

def test_OpticalPSF_aberrations_kwargs():
    """Test that OpticalPSF aberrations kwarg works just like specifying aberrations.
    """
    import time
    t1 = time.time()

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
        opt1.draw(scale=0.2*lod).array, opt2.draw(scale=0.2*lod).array,
        err_msg="Optical PSF depends on how aberrations are specified (4,8,11)")

    # Repeat with all aberrations up to index 11, using a regular list, not a numpy array
    opt1 = galsim.OpticalPSF(lod, defocus=.5, astig1=0.5, astig2=0.3, coma1=0.4, coma2=-0.3,
                             trefoil1=-0.2, trefoil2=0.1, spher=-0.8, obscuration=obscuration) 
    aberrations = [ 0.0 ] * 4 + [ 0.5, 0.5, 0.3, 0.4, -0.3, -0.2, 0.1, -0.8 ]
    opt2 = galsim.OpticalPSF(lod, obscuration=obscuration, aberrations=aberrations)
    np.testing.assert_array_equal(
        opt1.draw(scale=0.2*lod).array, opt2.draw(scale=0.2*lod).array,
        err_msg="Optical PSF depends on how aberrations are specified (full list)")

    # Also, check for proper response to weird inputs.
    try:
        # aberrations must be a list or an array
        np.testing.assert_raises(TypeError,galsim.OpticalPSF,lod,aberrations=0.3)
        # It must have at least 5 elements
        np.testing.assert_raises(ValueError,galsim.OpticalPSF,lod,aberrations=[0.0]*4)
        # It must (currently) have at most 12 elements
        np.testing.assert_raises(ValueError,galsim.OpticalPSF,lod,aberrations=[0.0]*15)
        if 'assert_warns' in np.testing.__dict__:
            # The first 4 elements must be 0. (Just a warning!)
            np.testing.assert_warns(UserWarning,galsim.OpticalPSF,lod,aberrations=[0.3]*8)
        # Cannot provide both aberrations and specific ones by name.
        np.testing.assert_raises(TypeError,galsim.OpticalPSF,lod,aberrations=np.zeros(8),
                                 defocus=-0.12)
    except ImportError:
        print 'The assert_raises tests require nose'


    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2 - t1)

def test_OpticalPSF_flux_scaling():
    """Test flux scaling for OpticalPSF.
    """
    import time
    t1 = time.time()

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
        obj.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __imul__.")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, pad_factor=test_pad_factor,
        defocus=test_defocus, astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj /= 2.
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __idiv__.")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, pad_factor=test_pad_factor,
        defocus=test_defocus, astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = obj * 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __rmul__ (result).")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, pad_factor=test_pad_factor,
        defocus=test_defocus, astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = 2. * obj
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux * 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __mul__ (result).")
    obj = galsim.OpticalPSF(
        lam_over_diam=test_loD, oversampling=test_oversampling, pad_factor=test_pad_factor,
        defocus=test_defocus, astig1=test_astig1, astig2=test_astig2, flux=test_flux)
    obj2 = obj / 2.
    # First test that original obj is unharmed... (also tests that .copy() is working)
    np.testing.assert_almost_equal(
        obj.getFlux(), test_flux, decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (original).")
    # Then test new obj2 flux
    np.testing.assert_almost_equal(
        obj2.getFlux(), test_flux / 2., decimal=param_decimal,
        err_msg="Flux param inconsistent after __div__ (result).")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_OpticalPSF_pupil_plane():
    """Test the ability to generate a PSF using an image of the pupil plane.
    """
    import time
    t1 = time.time()

    # Test case: lam/diam=0.12, obscuration=0.18, 4 struts of the default width and with rotation
    # from the vertical of -15 degrees.  There are two versions of these tests at different
    # oversampling levels.
    #
    # To generate the pupil plane that was saved for this case, I did the following:
    # - Temporarily edited galsim/optics.py right after the call to generate_pupil_plane() in the
    #   wavefront() method, adding the following lines:
    #   tmp_im = utilities.roll2d(in_pupil, (in_pupil.shape[0] / 2, in_pupil.shape[1] / 2))
    #   tmp_im = galsim.Image(np.ascontiguousarray(tmp_im).astype(np.int32))
    #   tmp_im.write('tests/Optics_comparison_images/sample_pupil_rolled.fits')
    # - Executed the following command:
    #   oversampling = 1.5
    #   pad_factor = 1.5
    #   galsim.OpticalPSF(0.12, obscuration=0.18, nstruts=4, strut_angle=-15.*galsim.degrees,
    #                     oversampling=oversampling, pad_factor=pad_factor)
    # - Then I made it write to
    #   tests/Optics_comparison_images/sample_pupil_rolled_oversample.fits.gz, and reran the command
    #   with oversampling = 4. and pad_factor = 4.
    #
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
    im = galsim.fits.read(os.path.join(imgdir, pp_file))
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
    im.scale = 1. # this doesn't matter, just put something so the next line works
    int_im = galsim.InterpolatedImage(galsim.Image(im, scale=im.scale, dtype=np.float32),
                                      calculate_maxk=False, calculate_stepk=False,
                                      x_interpolant='linear')
    new_im = int_im.drawImage(scale=rescale_fac*im.scale, method='no_pixel')
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
    sub_im = im[im.bounds.addBorder(remove_pad)]
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
    big_im = galsim.Image(im.bounds.addBorder(new_pad))
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
    test_psf_3 = galsim.OpticalPSF(
        lam_over_diam, obscuration=obscuration, oversampling=pp_oversampling,
        pupil_plane_im=os.path.join(imgdir, pp_file),
        pad_factor=pp_pad_factor)
    im_test_psf_3 = test_psf_3.drawImage(scale=scale)
    np.testing.assert_almost_equal(
        im_test_psf.array, im_test_psf_2.array, decimal=pp_decimal,
        err_msg="Inconsistent OpticalPSF image from Image vs. array.")
    np.testing.assert_almost_equal(
        im_test_psf.array, im_test_psf_3.array, decimal=pp_decimal,
        err_msg="Inconsistent OpticalPSF image from Image vs. file read-in.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_check_all_contiguous()
    test_simple_wavefront()
    test_simple_mtf()
    test_simple_ptf()
    test_consistency_psf_mtf()
    test_wavefront_image_view()
    test_psf_image_view()
    test_otf_image_view()
    test_mtf_image_view()
    test_ptf_image_view()
    test_OpticalPSF_flux()
    test_OpticalPSF_vs_Airy()
    test_OpticalPSF_vs_Airy_with_obs()
    test_OpticalPSF_aberrations_struts()
    test_OpticalPSF_aberrations_kwargs()
    test_OpticalPSF_flux_scaling()
    test_OpticalPSF_pupil_plane()
