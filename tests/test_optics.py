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
import numpy as np
import os
import sys

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


# If the parameter below is set True, then the function test_OpticalPSF_aberration() will run
# each aberration individually, as well as doing a multi-aberration regression test.  This adds
# ~10s to the test run time on a fast-ish laptop and is thus disabled by default.
RUN_ALL_SINGLE_ABERRATIONS = False


def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_check_all_contiguous():
    """Test all galsim.optics outputs are C-contiguous as required by the galsim.Image class.
    """
    import time
    t1 = time.time()
    # Check basic outputs from wavefront, psf and mtf (array contents won't matter, so we'll use
    # a pure circular pupil)
    assert galsim.optics.wavefront(array_shape=testshape).flags.c_contiguous
    assert galsim.optics.psf(array_shape=testshape).flags.c_contiguous
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
    wf = galsim.optics.wavefront(array_shape=testshape, dx=dx_test, lam_over_diam=lod_test)
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
    mtf = galsim.optics.mtf(array_shape=testshape, dx=dx_test, lam_over_diam=lod_test)
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
    psf = galsim.optics.psf(array_shape=testshape, dx=dx_test, lam_over_diam=lod_test)
    psf *= dx_test**2 # put the PSF into flux units rather than SB for comparison
    mtf_test = np.abs(np.fft.fft2(psf))
    # Compare
    mtf = galsim.optics.mtf(array_shape=testshape, dx=dx_test, lam_over_diam=lod_test)
    np.testing.assert_array_almost_equal(mtf, mtf_test, decimal=decimal_dft)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_wavefront_image_view():
    """Test that the ImageF.array view of the wavefront is consistent with the wavefront array.
    """
    import time
    t1 = time.time()
    array = galsim.optics.wavefront(array_shape=testshape)
    (real, imag) = galsim.optics.wavefront_image(array_shape=testshape)
    np.testing.assert_array_almost_equal(array.real.astype(np.float32), real.array, decimal)
    np.testing.assert_array_almost_equal(array.imag.astype(np.float32), imag.array, decimal)
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_psf_image_view():
    """Test that the ImageF.array view of the PSF is consistent with the PSF array.
    """
    import time
    t1 = time.time()
    array = galsim.optics.psf(array_shape=testshape)
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
        optics_test = galsim.OpticalPSF(lam_over_diam=lod, pad_factor=1)
        optics_array = optics_test.draw(dx=.25*lod, image=image).array 
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
        optics_test = galsim.OpticalPSF(lam_over_diam=lod, pad_factor=1)#pad same as an Airy, natch!
        airy_array = airy_test.draw(dx=.25*lod, image=image).array
        optics_array = optics_test.draw(dx=.25*lod, image=image).array 
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
        optics_test = galsim.OpticalPSF(lam_over_diam=lod, pad_factor=1, obscuration=obs)
        airy_array = airy_test.draw(dx=1.,image=image).array
        optics_array = optics_test.draw(dx=1.,image=image).array 
        np.testing.assert_array_almost_equal(optics_array, airy_array, decimal_dft, 
                err_msg="Unaberrated Optical with obscuration not quite equal to Airy")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_OpticalPSF_aberration():
    """Test the generation of optical aberration against a known result.
    """
    import time
    t1 = time.time()
    lod = 0.04
    obscuration = 0.3
    imsize = 128 # Size of saved images as generated by generate_optics_comparison_images.py
    myImg = galsim.ImageD(imsize, imsize)

    if RUN_ALL_SINGLE_ABERRATIONS:
        # test defocus
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_defocus.fits"))
        optics = galsim.OpticalPSF(lod, defocus=.5, obscuration=obscuration)
        myImg = optics.draw(myImg, dx=0.2*lod, use_true_center=False)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (defocus) disagrees with expected result")

        # test astig1
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_astig1.fits"))
        optics = galsim.OpticalPSF(lod, defocus=.5, astig1=.5, obscuration=obscuration)
        myImg = optics.draw(myImg, dx=0.2*lod, use_true_center=False)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (astig1) disagrees with expected result")

        # test astig2
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_astig2.fits"))
        optics = galsim.OpticalPSF(lod, defocus=.5, astig2=.5, obscuration=obscuration)
        myImg = optics.draw(myImg, dx=0.2*lod, use_true_center=False)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (astig2) disagrees with expected result")

        # test coma1
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_coma1.fits"))
        optics = galsim.OpticalPSF(lod, coma1=.5, obscuration=obscuration)
        myImg = optics.draw(myImg, dx=0.2*lod, use_true_center=False)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (coma1) disagrees with expected result")

        # test coma2
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_coma2.fits"))
        optics = galsim.OpticalPSF(lod, coma2=.5, obscuration=obscuration)
        myImg = optics.draw(myImg, dx=0.2*lod, use_true_center=False)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (coma2) disagrees with expected result")

        # test trefoil1
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_trefoil1.fits"))
        optics = galsim.OpticalPSF(lod, trefoil1=.5, obscuration=obscuration)
        myImg = optics.draw(myImg, dx=0.2*lod, use_true_center=False)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (trefoil1) disagrees with expected result")

        # test trefoil2
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_trefoil2.fits"))
        optics = galsim.OpticalPSF(lod, trefoil2=.5, obscuration=obscuration)
        myImg = optics.draw(myImg, dx=0.2*lod, use_true_center=False)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (trefoil2) disagrees with expected result")

        # test spherical
        savedImg = galsim.fits.read(os.path.join(imgdir, "optics_spher.fits"))
        optics = galsim.OpticalPSF(lod, spher=.5, obscuration=obscuration)
        myImg = optics.draw(myImg, dx=0.2*lod, use_true_center=False)
        np.testing.assert_array_almost_equal(
            myImg.array, savedImg.array, 6,
            err_msg="Optical aberration (spher) disagrees with expected result")

    # test all aberrations
    savedImg = galsim.fits.read(os.path.join(imgdir, "optics_all.fits"))
    optics = galsim.OpticalPSF(lod, defocus=.5, astig1=0.5, astig2=0.3, coma1=0.4, coma2=-0.3,
                               trefoil1=-0.2, trefoil2=0.1, spher=-0.8, obscuration=obscuration)
    myImg = optics.draw(myImg, dx=0.2*lod, use_true_center=False)
    np.testing.assert_array_almost_equal(
        myImg.array, savedImg.array, 6,
        err_msg="Optical aberration (all aberrations) disagrees with expected result")

    t2 = time.time()
    print 'time for %s = %.2f' % (funcname(), t2 - t1)

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
    test_OpticalPSF_aberration()
