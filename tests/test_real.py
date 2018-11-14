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

from __future__ import print_function
import numpy as np
import os
import sys

import galsim
from galsim_test_helpers import *


bppath = os.path.join(galsim.meta_data.share_dir, "bandpasses")
sedpath = os.path.join(galsim.meta_data.share_dir, "SEDs")

# set up any necessary info for tests
### Note: changes to either of the tests below might require regeneration of the catalog and image
### files that are saved here.  Modify with care!!!
image_dir = './real_comparison_images'
catalog_file = 'test_catalog.fits'

# some helper functions
def ellip_to_moments(e1, e2, sigma):
    a_val = (1.0 + e1) / (1.0 - e1)
    b_val = np.sqrt(a_val - (0.5*(1.0+a_val)*e2)**2)
    mxx = a_val * (sigma**2) / b_val
    myy = (sigma**2) / b_val
    mxy = 0.5 * e2 * (mxx + myy)
    return mxx, myy, mxy

def moments_to_ellip(mxx, myy, mxy):
    e1 = (mxx - myy) / (mxx + myy)
    e2 = 2*mxy / (mxx + myy)
    sig = (mxx*myy - mxy**2)**(0.25)
    return e1, e2, sig

@timer
def test_real_galaxy_catalog():
    """Test basic operations of RealGalaxyCatalog"""

    # Start with the test RGC that we will use throughout this test file.
    rgc = galsim.RealGalaxyCatalog(file_name=catalog_file, dir=image_dir)

    assert len(rgc) == rgc.nobjects == rgc.getNObjects() == 2
    assert rgc.file_name == os.path.join(image_dir, catalog_file)
    assert rgc.image_dir == image_dir

    print('sample = ',rgc.sample)
    print('ident = ',rgc.ident)
    assert rgc.sample == None
    assert len(rgc.ident) == 2

    gal1 = rgc.getGalImage(0)
    assert isinstance(gal1, galsim.Image)
    psf1 = rgc.getPSFImage(0)
    assert isinstance(psf1, galsim.Image)
    noise, scale, var = rgc.getNoiseProperties(0)
    assert noise is None  # No noise images for the test catalog.
    print('noise info = ',noise, scale, var)
    np.testing.assert_almost_equal(scale, 0.03)
    assert var < 1.e-5

    assert rgc.getIndexForID(100533) == 0

    # With _nobjects_only=True, it doesn't finish loadin
    rgc2 = galsim.RealGalaxyCatalog(file_name=catalog_file, dir=image_dir, _nobjects_only=True)
    assert len(rgc2) == rgc2.nobjects == rgc2.getNObjects() == 2
    assert rgc2.file_name == os.path.join(image_dir, catalog_file)
    assert rgc2.image_dir == image_dir
    assert rgc2.sample == None
    with assert_raises(AttributeError):
        rgc2.ident
    with assert_raises(AttributeError):
        rgc2.getGalImage(0)
    with assert_raises(AttributeError):
        rgc2.getPSFImage(0)

    assert_raises(TypeError, galsim.RealGalaxyCatalog, catalog_file, dir=image_dir, sample='25.2')
    assert_raises(ValueError, galsim.RealGalaxyCatalog, sample='23.2')
    assert_raises(ValueError, galsim.RealGalaxyCatalog, sample='23.2')
    assert_raises(OSError, galsim.RealGalaxyCatalog, file_name='invalid.fits')
    assert_raises(ValueError, rgc.getIndexForID, 1234)
    assert_raises(IndexError, rgc.getGalImage, 5)
    assert_raises(IndexError, rgc.getPSFImage, 5)
    assert_raises(IndexError, rgc.getNoiseProperties, 5)

    # The test catalog doesn't have noise information, so we use this hack to test the
    # behavior of another IndexError that would be raised in the usual case.
    rgc.noise_file_name = [ 'none' for i in rgc.ident ]
    assert_raises(IndexError, rgc.getNoiseProperties, 5)

    assert_raises(OSError, galsim.RealGalaxyCatalog, dir=image_dir)
    assert_raises(OSError, galsim.RealGalaxyCatalog, file_name='25.2.fits', dir=image_dir)
    assert_raises(OSError, galsim.RealGalaxyCatalog, file_name='23.5.fits', dir='invalid')

    # Test the catalog used by a few demos.
    rgc = galsim.RealGalaxyCatalog(sample='23.5_example', dir='../examples/data')
    assert(rgc.sample == '23.5_example')
    assert(len(rgc.ident) == 100)

    # Now test out the real ones.  But if they aren't installed, abort gracefully.
    try:
        rgc = galsim.RealGalaxyCatalog(sample='25.2')
    except OSError:
        print('Skipping tests of 25.2 sample, since not downloaded.')
    else:
        print('sample = ',rgc.sample)
        print('len(ident) = ',len(rgc.ident))
        assert(rgc.sample == '25.2')
        assert(len(rgc.ident) == 87798)

    try:
        rgc = galsim.RealGalaxyCatalog(sample='23.5')
    except OSError:
        print('Skipping tests of 25.2 sample, since not downloaded.')
    else:
        print('sample = ',rgc.sample)
        print('len(ident) = ',len(rgc.ident))
        assert(rgc.sample == '23.5')
        assert(len(rgc.ident) == 56062)

    # Check error message if COSMOS galaxies aren't in share_dir.  Do this by temporarily
    # changing share_dir value.
    save = galsim.meta_data.share_dir
    galsim.meta_data.share_dir = image_dir
    try:
        rgc = galsim.RealGalaxyCatalog(sample='23.5')
    except OSError as err:
        assert 'Run the program galsim_download_cosmos -s 23.5' in str(err)
    else:
        assert False, "Automatic sample=23.5 should have failed with share_dir = " + image_dir
    galsim.meta_data.share_dir = save


@timer
def test_real_galaxy_ideal():
    """Test accuracy of various calculations with fake Gaussian RealGalaxy vs. ideal expectations"""
    ind_fake = 1 # index of mock galaxy (Gaussian) in catalog
    fake_gal_fwhm = 0.7 # arcsec
    fake_gal_shear1 = 0.29 # shear representing intrinsic shape component 1
    fake_gal_shear2 = -0.21 # shear representing intrinsic shape component 2
    # note non-round, to detect possible issues with x<->y or others that might not show up using
    # circular galaxy

    fake_gal_flux = 1000.0
    fake_gal_orig_PSF_fwhm = 0.1 # arcsec
    fake_gal_orig_PSF_shear1 = 0.0
    fake_gal_orig_PSF_shear2 = -0.07

    # read in faked Gaussian RealGalaxy from file
    rgc = galsim.RealGalaxyCatalog(catalog_file, dir=image_dir)
    assert len(rgc) == rgc.getNObjects() == rgc.nobjects == len(rgc.cat)
    rg = galsim.RealGalaxy(rgc, index=ind_fake)
    # as a side note, make sure it behaves okay given a legit RNG and a bad RNG
    # or when trying to specify the galaxy too many ways
    rg_1 = galsim.RealGalaxy(rgc, index = ind_fake, rng = galsim.BaseDeviate(1234))
    rg_2 = galsim.RealGalaxy(rgc, random=True)

    assert_raises(TypeError, galsim.RealGalaxy, rgc, index=ind_fake, rng='foo')
    assert_raises(TypeError, galsim.RealGalaxy, rgc)
    assert_raises(TypeError, galsim.RealGalaxy, rgc, index=ind_fake, flux=12, flux_rescale=2)

    assert_raises(ValueError, galsim.RealGalaxy, rgc, index=ind_fake, id=0)
    assert_raises(ValueError, galsim.RealGalaxy, rgc, index=ind_fake, random=True)
    assert_raises(ValueError, galsim.RealGalaxy, rgc, id=0, random=True)

    # Different RNGs give different random galaxies.
    rg_3 = galsim.RealGalaxy(rgc, random=True, rng=galsim.BaseDeviate(12345))
    rg_4 = galsim.RealGalaxy(rgc, random=True, rng=galsim.BaseDeviate(67890))
    assert rg_3.index != rg_4.index, 'Different seeds did not give different random objects!'

    gsp = galsim.GSParams(xvalue_accuracy=1.e-8, kvalue_accuracy=1.e-8)
    rg_5 = galsim.RealGalaxy(rgc, random=True, rng=galsim.BaseDeviate(67890), gsparams=gsp)
    assert rg_5 != rg_4
    assert rg_5 == rg_4.withGSParams(gsp)

    check_basic(rg, "RealGalaxy", approx_maxsb=True)
    check_basic(rg_1, "RealGalaxy", approx_maxsb=True)
    check_basic(rg_2, "RealGalaxy", approx_maxsb=True)

    do_pickle(rgc, lambda x: [ x.getGalImage(ind_fake), x.getPSFImage(ind_fake),
                               x.getNoiseProperties(ind_fake) ])
    do_pickle(rgc, lambda x: drawNoise(x.getNoise(ind_fake,rng=galsim.BaseDeviate(123))))
    do_pickle(rgc)
    do_pickle(rg, lambda x: [ x.gal_image, x.psf_image, repr(x.noise),
                              x.original_psf.flux, x.original_gal.flux, x.flux ])
    do_pickle(rg, lambda x: x.drawImage(nx=20, ny=20, scale=0.7))
    do_pickle(rg_1, lambda x: x.drawImage(nx=20, ny=20, scale=0.7))
    do_pickle(rg)
    do_pickle(rg_1)

    ## for the generation of the ideal right answer, we need to add the intrinsic shape of the
    ## galaxy and the lensing shear using the rule for addition of distortions which is ugly, but oh
    ## well:
    targ_pixel_scale = [0.18, 0.25] # arcsec
    targ_PSF_fwhm = [0.7, 1.0] # arcsec
    targ_PSF_shear1 = [-0.03, 0.0]
    targ_PSF_shear2 = [0.05, -0.08]
    targ_applied_shear1 = 0.06
    targ_applied_shear2 = -0.04

    fwhm_to_sigma = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))

    (d1, d2) = galsim.utilities.g1g2_to_e1e2(fake_gal_shear1, fake_gal_shear2)
    (d1app, d2app) = galsim.utilities.g1g2_to_e1e2(targ_applied_shear1, targ_applied_shear2)
    denom = 1.0 + d1*d1app + d2*d2app
    dapp_sq = d1app**2 + d2app**2
    d1tot = (d1 + d1app + d2app/dapp_sq*(1.0 - np.sqrt(1.0-dapp_sq))*(d2*d1app - d1*d2app))/denom
    d2tot = (d2 + d2app + d1app/dapp_sq*(1.0 - np.sqrt(1.0-dapp_sq))*(d1*d2app - d2*d1app))/denom

    # convolve with a range of Gaussians, with and without shear (note, for this test all the
    # original and target ePSFs are Gaussian - there's no separate pixel response so that everything
    # can be calculated analytically)
    for tps in targ_pixel_scale:
        for tpf in targ_PSF_fwhm:
            for tps1 in targ_PSF_shear1:
                for tps2 in targ_PSF_shear2:
                    print('tps,tpf,tps1,tps2 = ',tps,tpf,tps1,tps2)
                    # make target PSF
                    targ_PSF = galsim.Gaussian(fwhm = tpf).shear(g1=tps1, g2=tps2)
                    # simulate image
                    tmp_gal = rg.withFlux(fake_gal_flux).shear(g1=targ_applied_shear1,
                                                               g2=targ_applied_shear2)
                    final_tmp_gal = galsim.Convolve(targ_PSF, tmp_gal)
                    sim_image = final_tmp_gal.drawImage(scale=tps, method='no_pixel')
                    # galaxy sigma, in units of pixels on the final image
                    sigma_ideal = (fake_gal_fwhm/tps)*fwhm_to_sigma
                    # compute analytically the expected galaxy moments:
                    mxx_gal, myy_gal, mxy_gal = ellip_to_moments(d1tot, d2tot, sigma_ideal)
                    # compute analytically the expected PSF moments:
                    targ_PSF_e1, targ_PSF_e2 = galsim.utilities.g1g2_to_e1e2(tps1, tps2)
                    targ_PSF_sigma = (tpf/tps)*fwhm_to_sigma
                    mxx_PSF, myy_PSF, mxy_PSF = ellip_to_moments(
                            targ_PSF_e1, targ_PSF_e2, targ_PSF_sigma)
                    # get expected e1, e2, sigma for the PSF-convolved image
                    tot_e1, tot_e2, tot_sigma = moments_to_ellip(
                            mxx_gal+mxx_PSF, myy_gal+myy_PSF, mxy_gal+mxy_PSF)

                    # compare with images that are expected
                    expected_gaussian = galsim.Gaussian(
                            flux = fake_gal_flux, sigma = tps*tot_sigma)
                    expected_gaussian = expected_gaussian.shear(e1 = tot_e1, e2 = tot_e2)
                    expected_image = galsim.ImageD(
                            sim_image.array.shape[0], sim_image.array.shape[1])
                    expected_gaussian.drawImage(expected_image, scale=tps, method='no_pixel')
                    printval(expected_image,sim_image)
                    np.testing.assert_array_almost_equal(
                        sim_image.array, expected_image.array, decimal = 3,
                        err_msg = "Error in comparison of ideal Gaussian RealGalaxy calculations")


@timer
def test_real_galaxy_makeFromImage():
    """Test accuracy of various calculations with fake Gaussian RealGalaxy vs. ideal expectations"""
    # read in faked Gaussian RealGalaxy from file
    rgc = galsim.RealGalaxyCatalog(catalog_file, dir=image_dir)
    rg = galsim.RealGalaxy(rgc, index=1)

    gal_image = rg.gal_image
    psf = rg.original_psf
    xi = rg.noise
    rg_2 = galsim.RealGalaxy.makeFromImage(gal_image, psf, xi)

    check_basic(rg_2, "RealGalaxy", approx_maxsb=True)
    do_pickle(rg_2, lambda x: [ x.gal_image, x.psf_image, repr(x.noise),
                                x.original_psf.flux, x.original_gal.flux, x.flux ])
    do_pickle(rg_2, lambda x: x.drawImage(nx=20, ny=20, scale=0.7))
    do_pickle(rg_2)

    # See if we get reasonably consistent results for rg and rg_2
    psf = galsim.Kolmogorov(fwhm=0.6)
    obj1 = galsim.Convolve(psf, rg)
    obj2 = galsim.Convolve(psf, rg_2)
    im1 = obj1.drawImage(scale=0.2, nx=12, ny=12)
    im2 = obj2.drawImage(image=im1.copy())
    atol = obj1.flux*3e-5
    np.testing.assert_allclose(im1.array, im2.array, rtol=0, atol=atol)


@timer
def test_real_galaxy_saved():
    """Test accuracy of various calculations with real RealGalaxy vs. stored SHERA result"""
    ind_real = 0 # index of real galaxy in catalog
    shera_file = 'real_comparison_images/shera_result.fits'
    shera_target_PSF_file = 'real_comparison_images/shera_target_PSF.fits'
    shera_target_pixel_scale = 0.24
    shera_target_flux = 1000.0

    # read in real RealGalaxy from file
    # rgc = galsim.RealGalaxyCatalog(catalog_file, dir=image_dir)
    # This is an alternate way to give the directory -- as part of the catalog file name.
    full_catalog_file = os.path.join(image_dir,catalog_file)
    rgc = galsim.RealGalaxyCatalog(full_catalog_file)
    rg = galsim.RealGalaxy(rgc, index=ind_real)

    # read in expected result for some shear
    shera_image = galsim.fits.read(shera_file)
    shera_target_PSF_image = galsim.fits.read(shera_target_PSF_file)
    shera_target_PSF_image.scale = shera_target_pixel_scale

    # simulate the same galaxy with GalSim
    targ_applied_shear1 = 0.06
    targ_applied_shear2 = -0.04
    tmp_gal = rg.withFlux(shera_target_flux).shear(g1=targ_applied_shear1,
                                                   g2=targ_applied_shear2)
    tmp_psf = galsim.InterpolatedImage(shera_target_PSF_image)
    tmp_gal = galsim.Convolve(tmp_gal, tmp_psf)
    sim_image = tmp_gal.drawImage(scale=shera_target_pixel_scale, method='no_pixel')

    # there are centroid issues when comparing Shera vs. SBProfile outputs, so compare 2nd moments
    # instead of images
    sbp_res = sim_image.FindAdaptiveMom()
    shera_res = shera_image.FindAdaptiveMom()

    np.testing.assert_almost_equal(sbp_res.observed_shape.e1,
                                   shera_res.observed_shape.e1, 2,
                                   err_msg = "Error in comparison with SHERA result: e1")
    np.testing.assert_almost_equal(sbp_res.observed_shape.e2,
                                   shera_res.observed_shape.e2, 2,
                                   err_msg = "Error in comparison with SHERA result: e2")
    np.testing.assert_almost_equal(sbp_res.moments_sigma, shera_res.moments_sigma, 2,
                                   err_msg = "Error in comparison with SHERA result: sigma")

    check_basic(rg, "RealGalaxy", approx_maxsb=True)

    # Check picklability
    do_pickle(rgc, lambda x: [ x.getGalImage(ind_real), x.getPSFImage(ind_real),
                               x.getNoiseProperties(ind_real) ])
    do_pickle(rgc, lambda x: drawNoise(x.getNoise(ind_real,rng=galsim.BaseDeviate(123))))
    do_pickle(rg, lambda x: galsim.Convolve([x,galsim.Gaussian(sigma=1.7)]).drawImage(
                                nx=20, ny=20, scale=0.7))
    do_pickle(rgc)
    do_pickle(rg)


@timer
def test_pickle_crg():
    """Just do some pickling tests of ChromaticRealGalaxy."""
    f606w_cat = galsim.RealGalaxyCatalog('AEGIS_F606w_catalog.fits', dir=image_dir)
    f814w_cat = galsim.RealGalaxyCatalog('AEGIS_F814w_catalog.fits', dir=image_dir)
    crg = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], index=0)

    do_pickle(crg)
    do_pickle(crg, lambda x: x.drawImage(f606w_cat.getBandpass()))

    # Check that missing band raises ValueError
    orig_band = f606w_cat.band
    f606w_cat.band = 'eggs'
    with assert_raises(ValueError):
        f606w_cat.getBandpass()
    f606w_cat.band = orig_band
    f606w_cat.getBandpass()


@timer
def test_crg_roundtrip():
    """Test that drawing a ChromaticRealGalaxy using the HST collecting area and filter gives back
    the original image.
    """
    f606w_cat = galsim.RealGalaxyCatalog('AEGIS_F606w_catalog.fits', dir=image_dir)
    f814w_cat = galsim.RealGalaxyCatalog('AEGIS_F814w_catalog.fits', dir=image_dir)

    indices = [0] if __name__ != "__main__" else list(range(len(f606w_cat)))

    for index in indices:
        orig_f606w = f606w_cat.getGalImage(index)
        orig_f814w = f814w_cat.getGalImage(index)

        crg = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], index=index)

        # Note that getPSF() return value already includes convolution by the pixel
        f606w_obj = galsim.Convolve(crg, f606w_cat.getPSF(index))
        f814w_obj = galsim.Convolve(crg, f814w_cat.getPSF(index))
        f606w = f606w_cat.getBandpass()
        f814w = f814w_cat.getBandpass()

        im_f606w = f606w_obj.drawImage(f606w, image=orig_f606w.copy(), method='no_pixel')
        im_f814w = f814w_obj.drawImage(f814w, image=orig_f814w.copy(), method='no_pixel')

        printval(im_f606w, orig_f606w)
        printval(im_f814w, orig_f814w)

        orig_f606w_mom = galsim.hsm.FindAdaptiveMom(orig_f606w)
        orig_f814w_mom = galsim.hsm.FindAdaptiveMom(orig_f814w)

        im_f606w_mom = galsim.hsm.FindAdaptiveMom(im_f606w)
        im_f814w_mom = galsim.hsm.FindAdaptiveMom(im_f814w)

        # Images are only pixel-by-pixel consistent to 5% or so.  However, if you actually plot the
        # residuals (which you can do by flipping False->True in printval in galsim_test_helpers),
        # they appear as ringing over the whole image.  Probably it's unrealistic to expect this
        # test to work perfectly since we're effectively deconvolving and then reconvolving by the
        # same PSF, not a fatter PSF.
        np.testing.assert_allclose(orig_f606w.array, im_f606w.array,
                                   rtol=0., atol=5e-2*orig_f606w.array.max())
        np.testing.assert_allclose(orig_f814w.array, im_f814w.array,
                                   rtol=0., atol=5e-2*orig_f814w.array.max())

        # Happily, the pixel-by-pixel residuals don't appear to affect the moments much:
        np.testing.assert_allclose(orig_f606w_mom.moments_amp,
                                   im_f606w_mom.moments_amp,
                                   rtol=1e-3, atol=0)
        np.testing.assert_allclose(orig_f606w_mom.moments_centroid.x,
                                   im_f606w_mom.moments_centroid.x,
                                   rtol=0., atol=1e-2)
        np.testing.assert_allclose(orig_f606w_mom.moments_centroid.y,
                                   im_f606w_mom.moments_centroid.y,
                                   rtol=0., atol=1e-2)
        np.testing.assert_allclose(orig_f606w_mom.moments_sigma,
                                   im_f606w_mom.moments_sigma,
                                   rtol=1e-3, atol=0)
        np.testing.assert_allclose(orig_f606w_mom.observed_shape.g1,
                                   im_f606w_mom.observed_shape.g1,
                                   rtol=0, atol=2e-4)
        np.testing.assert_allclose(orig_f606w_mom.observed_shape.g2,
                                   im_f606w_mom.observed_shape.g2,
                                   rtol=0, atol=2e-4)

        np.testing.assert_allclose(orig_f814w_mom.moments_amp,
                                   im_f814w_mom.moments_amp,
                                   rtol=1e-3, atol=0)
        np.testing.assert_allclose(orig_f814w_mom.moments_centroid.x,
                                   im_f814w_mom.moments_centroid.x,
                                   rtol=0., atol=1e-2)
        np.testing.assert_allclose(orig_f814w_mom.moments_centroid.y,
                                   im_f814w_mom.moments_centroid.y,
                                   rtol=0., atol=1e-2)
        np.testing.assert_allclose(orig_f814w_mom.moments_sigma,
                                   im_f814w_mom.moments_sigma,
                                   rtol=1e-3, atol=0)
        np.testing.assert_allclose(orig_f814w_mom.observed_shape.g1,
                                   im_f814w_mom.observed_shape.g1,
                                   rtol=0, atol=1e-4)
        np.testing.assert_allclose(orig_f814w_mom.observed_shape.g2,
                                   im_f814w_mom.observed_shape.g2,
                                   rtol=0, atol=1e-4)

    # Check some errors
    cats = [f606w_cat, f814w_cat]
    assert_raises(TypeError, galsim.ChromaticRealGalaxy, real_galaxy_catalogs=cats)
    assert_raises(TypeError, galsim.ChromaticRealGalaxy, cats, index=3, id=4)
    assert_raises(TypeError, galsim.ChromaticRealGalaxy, cats, index=3, random=True)
    assert_raises(TypeError, galsim.ChromaticRealGalaxy, cats, id=4, random=True)
    assert_raises(TypeError, galsim.ChromaticRealGalaxy, cats, random=True, rng='foo')


@timer
def test_crg_roundtrip_larger_target_psf():
    """Test that drawing a chromatic galaxy with a color gradient directly using an LSST-size PSF
    is equivalent to first drawing the galaxy to HST-like images, and then using ChromaticRealGalaxy
    to produce an LSST-like image.
    """
    # load some spectra
    bulge_SED = (galsim.SED(os.path.join(sedpath, 'CWW_E_ext.sed'), wave_type='ang',
                            flux_type='flambda')
                 .thin(rel_err=1e-3)
                 .withFluxDensity(target_flux_density=0.3, wavelength=500.0))

    disk_SED = (galsim.SED(os.path.join(sedpath, 'CWW_Sbc_ext.sed'), wave_type='ang',
                           flux_type='flambda')
                .thin(rel_err=1e-3)
                .withFluxDensity(target_flux_density=0.3, wavelength=500.0))

    bulge = galsim.Sersic(n=4, half_light_radius=0.6)*bulge_SED
    disk = galsim.Sersic(n=1, half_light_radius=0.4)*disk_SED
    # Decenter components a bit to make the test more complicated
    disk = disk.shift(0.05, 0.1)
    gal = (bulge+disk).shear(g1=0.3, g2=0.1)

    # Much faster to just use some achromatic HST-like PSFs.  We'll make them slightly different in
    # each band though.
    f606w_PSF = galsim.ChromaticObject(galsim.Gaussian(half_light_radius=0.05))
    f814w_PSF = galsim.ChromaticObject(galsim.Gaussian(half_light_radius=0.07))
    LSSTPSF = galsim.ChromaticAtmosphere(galsim.Kolmogorov(fwhm=0.7),
                                         600.0,
                                         zenith_angle=0.0*galsim.degrees)

    f606w = galsim.Bandpass(os.path.join(bppath, "ACS_wfc_F606W.dat"), 'nm').truncate()
    f814w = galsim.Bandpass(os.path.join(bppath, "ACS_wfc_F814W.dat"), 'nm')
    LSST_i = galsim.Bandpass(os.path.join(bppath, "LSST_r.dat"), 'nm')

    truth_image = galsim.Convolve(LSSTPSF, gal).drawImage(LSST_i, nx=24, ny=24, scale=0.2)
    f606w_image = galsim.Convolve(f606w_PSF, gal).drawImage(f606w, nx=192, ny=192, scale=0.03)
    f814w_image = galsim.Convolve(f814w_PSF, gal).drawImage(f814w, nx=192, ny=192, scale=0.03)

    crg = galsim.ChromaticRealGalaxy.makeFromImages(
            images=[f606w_image, f814w_image],
            bands=[f606w, f814w],
            PSFs=[f606w_PSF, f814w_PSF],
            xis=[galsim.UncorrelatedNoise(1e-16)]*2,
            SEDs=[bulge_SED, disk_SED])

    test_image = galsim.Convolve(crg, LSSTPSF).drawImage(LSST_i, nx=24, ny=24, scale=0.2)

    truth_mom = galsim.hsm.FindAdaptiveMom(truth_image)
    test_mom = galsim.hsm.FindAdaptiveMom(test_image)

    np.testing.assert_allclose(test_mom.moments_amp,
                               truth_mom.moments_amp,
                               rtol=1e-3, atol=0)
    np.testing.assert_allclose(test_mom.moments_centroid.x,
                               truth_mom.moments_centroid.x,
                               rtol=0., atol=1e-2)
    np.testing.assert_allclose(test_mom.moments_centroid.y,
                               truth_mom.moments_centroid.y,
                               rtol=0., atol=1e-2)
    np.testing.assert_allclose(test_mom.moments_sigma,
                               truth_mom.moments_sigma,
                               rtol=1e-3, atol=0)
    np.testing.assert_allclose(test_mom.observed_shape.g1,
                               truth_mom.observed_shape.g1,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(test_mom.observed_shape.g2,
                               truth_mom.observed_shape.g2,
                               rtol=0, atol=1e-4)


    # Invalid arguments
    with assert_raises(ValueError):
        crg = galsim.ChromaticRealGalaxy.makeFromImages(
            images=[f606w_image],
            bands=[f606w, f814w],
            PSFs=[f606w_PSF, f814w_PSF],
            xis=[galsim.UncorrelatedNoise(1e-16)]*2,
            SEDs=[bulge_SED, disk_SED])
    with assert_raises(ValueError):
        crg = galsim.ChromaticRealGalaxy.makeFromImages(
            images=[f606w_image, f814w_image],
            bands=[f606w],
            PSFs=[f606w_PSF, f814w_PSF],
            xis=[galsim.UncorrelatedNoise(1e-16)]*2,
            SEDs=[bulge_SED, disk_SED])
    with assert_raises(ValueError):
        crg = galsim.ChromaticRealGalaxy.makeFromImages(
            images=[f606w_image, f814w_image],
            bands=[f606w, f814w],
            PSFs=[f606w_PSF],
            xis=[galsim.UncorrelatedNoise(1e-16)]*2,
            SEDs=[bulge_SED, disk_SED])
    with assert_raises(ValueError):
        crg = galsim.ChromaticRealGalaxy.makeFromImages(
            images=[f606w_image, f814w_image],
            bands=[f606w, f814w],
            PSFs=[f606w_PSF, f814w_PSF],
            xis=[galsim.UncorrelatedNoise(1e-16)],
            SEDs=[bulge_SED, disk_SED])
    with assert_raises(ValueError):
        crg = galsim.ChromaticRealGalaxy.makeFromImages(
            images=[f606w_image, f814w_image],
            bands=[f606w, f814w],
            PSFs=[f606w_PSF, f814w_PSF],
            xis=[galsim.UncorrelatedNoise(1e-16)]*2,
            SEDs=[bulge_SED, disk_SED, disk_SED])


@timer
def test_ne():
    """ Check that inequality works as expected."""
    rgc = galsim.RealGalaxyCatalog(catalog_file, dir=image_dir)
    f606w_cat = galsim.RealGalaxyCatalog('AEGIS_F606w_catalog.fits', dir=image_dir)
    f814w_cat = galsim.RealGalaxyCatalog('AEGIS_F814w_catalog.fits', dir=image_dir)
    crg1 = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], index=0)
    crg2 = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], index=1)
    covspec1 = crg1.covspec
    covspec2 = crg2.covspec

    gsp = galsim.GSParams(folding_threshold=1.1e-3)

    objs = [galsim.RealGalaxy(rgc, index=0),
            galsim.RealGalaxy(rgc, index=1),
            galsim.RealGalaxy(rgc, index=0, x_interpolant='Linear'),
            galsim.RealGalaxy(rgc, index=0, k_interpolant='Linear'),
            galsim.RealGalaxy(rgc, index=0, flux=1.1),
            galsim.RealGalaxy(rgc, index=0, flux_rescale=1.2),
            galsim.RealGalaxy(rgc, index=0, area_norm=2),
            galsim.RealGalaxy(rgc, index=0, pad_factor=1.1),
            galsim.RealGalaxy(rgc, index=0, noise_pad_size=5.0),
            galsim.RealGalaxy(rgc, index=0, gsparams=gsp),
            crg1,
            crg2,
            galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], index=0, k_interpolant='Linear'),
            covspec1,
            covspec2]
    all_obj_diff(objs)
    for obj in objs[:-5]:
        do_pickle(obj)

    # CovarianceSpectrum and ChromaticRealGalaxy are both reprable, but their reprs are rather
    # large, so the eval(repr) checks take a long time.
    # Therefore, run them from command line, but not from pytest.
    if __name__ == '__main__':
        do_pickle(crg1)
        do_pickle(covspec1)
    else:
        do_pickle(crg1, irreprable=True)
        do_pickle(covspec1, irreprable=True)

@timer
def test_noise():
    """Check consistency of noise-related routines."""
    # The RealGalaxyCatalog.getNoise() routine should be tested to ensure consistency of results
    # with the getNoiseProperties() routine.  The former cannot be used across processes, but might
    # be used when running on a single processor, so we should make sure it gives proper output.
    # Need to use a real RealGalaxyCatalog with non-trivial noise correlation function.
    real_gal_dir = os.path.join('..','examples','data')
    real_gal_cat = 'real_galaxy_catalog_23.5_example.fits'
    real_cat = galsim.RealGalaxyCatalog(
        dir=real_gal_dir, file_name=real_gal_cat, preload=True)

    test_seed=987654
    test_index = 17
    cf_1 = real_cat.getNoise(test_index, rng=galsim.BaseDeviate(test_seed))
    im_2, pix_scale_2, var_2 = real_cat.getNoiseProperties(test_index)
    # Check the variance:
    var_1 = cf_1.getVariance()
    assert var_1==var_2,'Inconsistent noise variance from getNoise and getNoiseProperties'
    # Check the image:
    ii = galsim.InterpolatedImage(im_2, normalization='sb', calculate_stepk=False,
                                  calculate_maxk=False, x_interpolant='linear')
    cf_2 = galsim.correlatednoise._BaseCorrelatedNoise(galsim.BaseDeviate(test_seed), ii, im_2.wcs)
    cf_2 = cf_2.withVariance(var_2)
    assert cf_1==cf_2,'Inconsistent noise properties from getNoise and getNoiseProperties'


@timer
def test_area_norm():
    """Check that area_norm works as expected"""
    f606w_cat = galsim.RealGalaxyCatalog('AEGIS_F606w_catalog.fits', dir=image_dir)
    f814w_cat = galsim.RealGalaxyCatalog('AEGIS_F814w_catalog.fits', dir=image_dir)

    psf = galsim.Gaussian(fwhm=0.6)

    rng = galsim.BaseDeviate(5772)
    crg1 = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], random=True, rng=rng.duplicate())
    crg2 = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], random=True, rng=rng.duplicate(),
                                      area_norm=galsim.real.HST_area)
    assert crg1 != crg2
    LSST_i = galsim.Bandpass(os.path.join(bppath, "LSST_r.dat"), 'nm')
    obj1 = galsim.Convolve(crg1, psf)
    obj2 = galsim.Convolve(crg2, psf)
    im1 = obj1.drawImage(LSST_i, exptime=1, area=1)
    im2 = obj2.drawImage(LSST_i, exptime=1, area=galsim.real.HST_area)
    printval(im1, im2)
    np.testing.assert_array_almost_equal(im1.array, im2.array)
    np.testing.assert_almost_equal(
            obj1.noise.getVariance(),
            obj2.noise.getVariance() * galsim.real.HST_area**2)

    # area_norm is equivalant to an overall scaling
    crg3 = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], random=True, rng=rng.duplicate())
    crg3 /= galsim.real.HST_area
    obj3 = galsim.Convolve(crg3, psf)
    im3 = obj3.drawImage(LSST_i, exptime=1, area=galsim.real.HST_area)
    np.testing.assert_array_almost_equal(im3.array, im2.array)
    np.testing.assert_almost_equal(obj3.noise.getVariance(), obj2.noise.getVariance())

    rg1 = galsim.RealGalaxy(f606w_cat, index=1)
    rg2 = galsim.RealGalaxy(f606w_cat, index=1, area_norm=galsim.real.HST_area)
    assert rg1 != rg2
    obj1 = galsim.Convolve(rg1, psf)
    obj2 = galsim.Convolve(rg2, psf)
    im1 = obj1.drawImage()
    im2 = obj2.drawImage(exptime=1, area=galsim.real.HST_area)
    printval(im1, im2)
    np.testing.assert_array_almost_equal(im1.array, im2.array)
    np.testing.assert_almost_equal(
            obj1.noise.getVariance(),
            obj2.noise.getVariance() * galsim.real.HST_area**2)

    # area_norm is equivalant to an overall scaling
    rg3 = galsim.RealGalaxy(f606w_cat, index=1)
    rg3 /= galsim.real.HST_area
    obj3 = galsim.Convolve(rg3, psf)
    im3 = obj3.drawImage(exptime=1, area=galsim.real.HST_area)
    np.testing.assert_array_almost_equal(im3.array, im2.array)
    np.testing.assert_almost_equal(obj3.noise.getVariance(), obj2.noise.getVariance())



@timer
def test_crg_noise_draw_transform_commutativity():
    """Test commutativity of ChromaticRealGalaxy correlated noise under operations of drawImage and
    applying transformations.
    """
    LSST_i = galsim.Bandpass(os.path.join(bppath, "LSST_r.dat"), 'nm')
    f606w_cat = galsim.RealGalaxyCatalog('AEGIS_F606w_catalog.fits', dir=image_dir)
    f814w_cat = galsim.RealGalaxyCatalog('AEGIS_F814w_catalog.fits', dir=image_dir)

    psf = galsim.Gaussian(fwhm=0.6)
    crg = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], id=14886,
                                     maxk=psf.maxk)

    factor = 1.5
    g1 = g2 = 0.1
    mu = 1.2
    theta = 45*galsim.degrees
    jac = [1.1, 0.1, -0.1, 1.2]

    orig = galsim.Convolve(crg, psf)
    orig.drawImage(LSST_i)

    draw_transform_img = galsim.ImageD(16, 16, scale=0.2)
    transform_draw_img = draw_transform_img.copy()

    multiplied = orig * factor
    multiplied.drawImage(LSST_i) # needed to populate noise property
    (orig.noise*factor**2).drawImage(image=draw_transform_img)
    multiplied.noise.drawImage(image=transform_draw_img)
    np.testing.assert_array_almost_equal(
            draw_transform_img.array,
            transform_draw_img.array)

    divided = orig / factor
    divided.drawImage(LSST_i)
    (orig.noise/factor**2).drawImage(image=draw_transform_img)
    divided.noise.drawImage(image=transform_draw_img)
    np.testing.assert_array_almost_equal(
            draw_transform_img.array,
            transform_draw_img.array)

    expanded = orig.expand(factor)
    expanded.drawImage(LSST_i)
    orig.noise.expand(factor).drawImage(image=draw_transform_img)
    expanded.noise.drawImage(image=transform_draw_img)
    np.testing.assert_array_almost_equal(
            draw_transform_img.array,
            transform_draw_img.array)

    dilated = orig.dilate(factor)
    dilated.drawImage(LSST_i)
    orig.noise.dilate(factor).drawImage(image=draw_transform_img)
    dilated.noise.drawImage(image=transform_draw_img)
    np.testing.assert_array_almost_equal(
            draw_transform_img.array,
            transform_draw_img.array)

    magnified = orig.magnify(mu)
    magnified.drawImage(LSST_i)
    orig.noise.magnify(mu).drawImage(image=draw_transform_img)
    magnified.noise.drawImage(image=transform_draw_img)
    np.testing.assert_array_almost_equal(
            draw_transform_img.array,
            transform_draw_img.array)

    lensed = orig.lens(g1, g2, mu)
    lensed.drawImage(LSST_i)
    orig.noise.lens(g1, g2, mu).drawImage(image=draw_transform_img)
    lensed.noise.drawImage(image=transform_draw_img)
    np.testing.assert_array_almost_equal(
            draw_transform_img.array,
            transform_draw_img.array)

    rotated = orig.rotate(theta)
    rotated.drawImage(LSST_i)
    orig.noise.rotate(theta).drawImage(image=draw_transform_img)
    rotated.noise.drawImage(image=transform_draw_img)
    np.testing.assert_array_almost_equal(
            draw_transform_img.array,
            transform_draw_img.array)

    sheared = orig.shear(g1=g1, g2=g2)
    sheared.drawImage(LSST_i)
    orig.noise.shear(g1=g1, g2=g2).drawImage(image=draw_transform_img)
    sheared.noise.drawImage(image=transform_draw_img)
    np.testing.assert_array_almost_equal(
            draw_transform_img.array,
            transform_draw_img.array)

    transformed = orig.transform(*jac)
    transformed.drawImage(LSST_i)
    orig.noise.transform(*jac).drawImage(image=draw_transform_img)
    transformed.noise.drawImage(image=transform_draw_img)
    np.testing.assert_array_almost_equal(
            draw_transform_img.array,
            transform_draw_img.array)


def check_crg_noise(n_sed, n_im, n_trial, tol):
    print("Checking CRG noise for")
    print("n_sed = {}".format(n_sed))
    print("n_im = {}".format(n_im))
    print("n_trial = {}".format(n_trial))
    print("Constructing chromatic PSFs")
    in_PSF = galsim.ChromaticAiry(lam=700., diam=2.4)
    out_PSF = galsim.ChromaticAiry(lam=700., diam=0.6)

    print("Constructing filters and SEDs")
    waves = np.arange(550.0, 900.1, 10.0)
    visband = galsim.Bandpass(galsim.LookupTable(waves, np.ones_like(waves), interpolant='linear'),
                              wave_type='nm')
    split_points = np.linspace(550.0, 900.0, n_im+1, endpoint=True)
    bands = [visband.truncate(blue_limit=blim, red_limit=rlim)
             for blim, rlim in zip(split_points[:-1], split_points[1:])]

    maxk = max([out_PSF.evaluateAtWavelength(waves[0]).maxk,
                out_PSF.evaluateAtWavelength(waves[-1]).maxk])

    SEDs = [galsim.SED(galsim.LookupTable(waves, waves**i, interpolant='linear'),
                       flux_type='fphotons', wave_type='nm').withFlux(1.0, visband)
            for i in range(n_sed)]

    print("Constructing input noise correlation functions")
    rng = galsim.BaseDeviate(57721)
    in_xis = [galsim.getCOSMOSNoise(cosmos_scale=0.03, rng=rng)
              .dilate(1 + i * 0.05)
              .rotate(5 * i * galsim.degrees)
              for i in range(n_im)]

    print("Creating noise images")
    img_sets = []
    for i in range(n_trial):
        imgs = []
        for xi in in_xis:
            img = galsim.Image(128, 128, scale=0.03)
            img.addNoise(xi)
            imgs.append(img)
        img_sets.append(imgs)

    print("Constructing `ChromaticRealGalaxy`s")
    crgs = []
    for imgs in img_sets:
        crgs.append(galsim.ChromaticRealGalaxy.makeFromImages(
                imgs, bands, in_PSF, in_xis, SEDs=SEDs, maxk=maxk))

    print("Convolving by output PSF")
    objs = [galsim.Convolve(crg, out_PSF) for crg in crgs]

    with assert_raises(galsim.GalSimError):
        noise = objs[0].noise  # Invalid before drawImage is called

    print("Drawing through output filter")
    out_imgs = [obj.drawImage(visband, nx=30, ny=30, scale=0.1)
                for obj in objs]

    noise = objs[0].noise

    print("Measuring images' correlation functions")
    xi_obs = galsim.correlatednoise.CorrelatedNoise(out_imgs[0])
    for img in out_imgs[1:]:
        xi_obs += galsim.correlatednoise.CorrelatedNoise(img)
    xi_obs /= n_trial
    xi_obs_img = galsim.Image(30, 30, scale=0.1)
    xi_obs.drawImage(xi_obs_img)
    noise_img = galsim.Image(30, 30, scale=0.1)
    noise.drawImage(noise_img)

    print("Predicted/Observed variance:", noise.getVariance()/xi_obs.getVariance())
    print("Predicted/Observed xlag-1 covariance:", noise_img.array[14, 15]/xi_obs_img.array[14, 15])
    print("Predicted/Observed ylag-1 covariance:", noise_img.array[15, 14]/xi_obs_img.array[15, 14])
    # Just test that the covariances for nearest neighbor pixels are accurate.
    np.testing.assert_allclose(
            noise_img.array[14:17, 14:17], xi_obs_img.array[14:17, 14:17],
            rtol=0, atol=noise.getVariance()*tol)


@timer
def test_crg_noise():
    """Verify that we can propagate the noise covariance by actually measuring the covariance of
    some pure noise fields put through ChromaticRealGalaxy.
    """
    if __name__ == '__main__':
        check_crg_noise(2, 2, 50, tol=0.03)
        check_crg_noise(2, 3, 25, tol=0.03)
        check_crg_noise(3, 3, 25, tol=0.03)
    else:
        check_crg_noise(2, 2, 10, tol=0.05)


@timer
def test_crg_noise_pad():
    f606w_cat = galsim.RealGalaxyCatalog('AEGIS_F606w_catalog.fits', dir=image_dir)
    f814w_cat = galsim.RealGalaxyCatalog('AEGIS_F814w_catalog.fits', dir=image_dir)

    # If we don't use noise_pad_size, then when we draw an image larger than the original postage
    # stamp, it gets padded with (nearly) zeros.  We can check this by measuring the variance around
    # the edge of the image (so away from the galaxy light).
    crg = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], index=0)
    psf = galsim.Gaussian(fwhm=0.4)
    obj = galsim.Convolve(crg, psf)
    bandpass = f606w_cat.getBandpass()
    img = obj.drawImage(bandpass, nx=24, ny=24, scale=0.2)

    x = np.arange(24)
    x, y = np.meshgrid(x, x)
    edge = (x < 4) | (x > 19) | (y < 4) | (y > 19)
    print(np.var(img.array[edge]))
    edgevar = np.var(img.array[edge])
    np.testing.assert_allclose(edgevar, 0.0, rtol=0, atol=1e-11)

    # If we turn up noise_pad_size though, then the variance of the edge should match the variance
    # computed via CRG
    rng = galsim.BaseDeviate(577)
    crg = galsim.ChromaticRealGalaxy([f606w_cat, f814w_cat], index=0, rng=rng, noise_pad_size=4)
    obj = galsim.Convolve(crg, psf)
    img = obj.drawImage(bandpass, nx=24, ny=24, scale=0.2)
    edgevar = np.var(img.array[edge])
    print("expected variance: ", obj.noise.getVariance())
    print("edge variance: ", edgevar)
    # Not super accurate, but since we only have a handful of correlated pixels to use, that
    # may be expected.  More detailed tests of noise in test_crg_noise() show better accuracy.
    np.testing.assert_allclose(obj.noise.getVariance(), edgevar, atol=0, rtol=0.3)


if __name__ == "__main__":
    test_real_galaxy_catalog()
    test_real_galaxy_ideal()
    test_real_galaxy_saved()
    test_real_galaxy_makeFromImage()
    test_pickle_crg()
    test_crg_roundtrip()
    test_crg_roundtrip_larger_target_psf()
    test_ne()
    test_noise()
    test_area_norm()
    test_crg_noise_draw_transform_commutativity()
    test_crg_noise()
    test_crg_noise_pad()
