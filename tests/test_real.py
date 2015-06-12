# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))
try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

# set up any necessary info for tests
### Note: changes to either of the tests below might require regeneration of the catalog and image
### files that are saved here.  Modify with care!!!
image_dir = './real_comparison_images'
catalog_file = os.path.join(image_dir,'test_catalog.fits')

ind_fake = 1 # index of mock galaxy (Gaussian) in catalog
fake_gal_fwhm = 0.7 # arcsec
fake_gal_shear1 = 0.29 # shear representing intrinsic shape component 1
fake_gal_shear2 = -0.21 # shear representing intrinsic shape component 2; note non-round, to detect
              # possible issues with x<->y or others that might not show up using circular galaxy
fake_gal_flux = 1000.0
fake_gal_orig_PSF_fwhm = 0.1 # arcsec
fake_gal_orig_PSF_shear1 = 0.0
fake_gal_orig_PSF_shear2 = -0.07

targ_pixel_scale = [0.18, 0.25] # arcsec
targ_PSF_fwhm = [0.7, 1.0] # arcsec
targ_PSF_shear1 = [-0.03, 0.0]
targ_PSF_shear2 = [0.05, -0.08]
targ_applied_shear1 = 0.06
targ_applied_shear2 = -0.04

sigma_to_fwhm = 2.0*np.sqrt(2.0*np.log(2.0)) # multiply sigma by this to get FWHM for Gaussian
fwhm_to_sigma = 1.0/sigma_to_fwhm

ind_real = 0 # index of real galaxy in catalog
shera_file = 'real_comparison_images/shera_result.fits'
shera_target_PSF_file = 'real_comparison_images/shera_target_PSF.fits'
shera_target_pixel_scale = 0.24
shera_target_flux = 1000.0

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

def test_real_galaxy_ideal():
    """Test accuracy of various calculations with fake Gaussian RealGalaxy vs. ideal expectations"""
    import time
    t1 = time.time()
    # read in faked Gaussian RealGalaxy from file
    rgc = galsim.RealGalaxyCatalog(catalog_file, image_dir)
    rg = galsim.RealGalaxy(rgc, index = ind_fake)
    # as a side note, make sure it behaves okay given a legit RNG and a bad RNG
    # or when trying to specify the galaxy too many ways
    rg_1 = galsim.RealGalaxy(rgc, index = ind_fake, rng = galsim.BaseDeviate(1234))
    rg_2 = galsim.RealGalaxy(rgc, random=True)
    try:
        np.testing.assert_raises(TypeError, galsim.RealGalaxy, rgc, index=ind_fake, rng='foo')
        np.testing.assert_raises(AttributeError, galsim.RealGalaxy, rgc, index=ind_fake, id=0)
        np.testing.assert_raises(AttributeError, galsim.RealGalaxy, rgc, index=ind_fake, random=True)
        np.testing.assert_raises(AttributeError, galsim.RealGalaxy, rgc, id=0, random=True)
        np.testing.assert_raises(AttributeError, galsim.RealGalaxy, rgc)
    except ImportError:
        print 'The assert_raises tests require nose'

    do_pickle(rgc, lambda x: [ x.getGal(ind_fake), x.getPSF(ind_fake),
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
                    print 'tps,tpf,tps1,tps2 = ',tps,tpf,tps1,tps2
                    # make target PSF
                    targ_PSF = galsim.Gaussian(fwhm = tpf).shear(g1=tps1, g2=tps2)
                    # simulate image
                    sim_image = galsim.simReal(
                            rg, targ_PSF, tps, 
                            g1 = targ_applied_shear1, g2 = targ_applied_shear2,
                            rand_rotate = False, target_flux = fake_gal_flux)
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

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_real_galaxy_saved():
    """Test accuracy of various calculations with real RealGalaxy vs. stored SHERA result"""
    import time
    t1 = time.time()
    # read in real RealGalaxy from file
    rgc = galsim.RealGalaxyCatalog(catalog_file, image_dir)
    rg = galsim.RealGalaxy(rgc, index = ind_real)

    # read in expected result for some shear
    shera_image = galsim.fits.read(shera_file)
    shera_target_PSF_image = galsim.fits.read(shera_target_PSF_file)

    # simulate the same galaxy with GalSim
    sim_image = galsim.simReal(rg, shera_target_PSF_image, shera_target_pixel_scale,
                               g1 = targ_applied_shear1, g2 = targ_applied_shear2,
                               rand_rotate = False, target_flux = shera_target_flux)

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

    # Check picklability
    do_pickle(rgc, lambda x: [ x.getGal(ind_real), x.getPSF(ind_real),
                               x.getNoiseProperties(ind_real) ])
    do_pickle(rgc, lambda x: drawNoise(x.getNoise(ind_real,rng=galsim.BaseDeviate(123))))
    do_pickle(rg, lambda x: galsim.Convolve([x,galsim.Gaussian(sigma=1.7)]).drawImage(
                                nx=20, ny=20, scale=0.7))
    do_pickle(rgc)
    do_pickle(rg)

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_chromatic_real_galaxy():
    """Use some simplified simulated HST-like observations around r and i band to predict
    Euclid-ish visual band observations."""

    print "Constructing simplified HST PSF"
    HST_PSF = galsim.ChromaticAiry(lam=700, diam=2.4)

    print "Constructing simplified Euclid PSF"
    Euclid_PSF = galsim.ChromaticAiry(lam=700, diam=1.2)

    print "Constructing simple filters and SEDs"
    waves = np.arange(550.0, 825.1, 1.0)

    # Construct some simple filters.
    visband = galsim.Bandpass(galsim.LookupTable(waves, np.ones_like(waves), interpolant='linear'))
    rband = visband.truncate(blue_limit=550.0, red_limit=700.0)
    iband = visband.truncate(blue_limit=700.0, red_limit=825.0)

    const_SED = (galsim.SED(galsim.LookupTable(waves, np.ones_like(waves),
                                               interpolant='linear'))
                 .withFluxDensity(1.0, 700.0))
    linear_SED = (galsim.SED(galsim.LookupTable(waves, (waves-550.0)/(825-550),
                                                interpolant='linear'))
                  .withFluxDensity(1.0, 700.0))

    print "Constructing galaxy"
    gal1 = galsim.Gaussian(half_light_radius=0.45).shear(e1=0.1, e2=0.2).shift(0.1, 0.2)
    gal2 = galsim.Gaussian(half_light_radius=0.35).shear(e1=-0.1, e2=0.4).shift(-0.3, 0.5)
    gal = gal1 * const_SED + gal2 * linear_SED

    HST_prof = galsim.Convolve(gal, HST_PSF)
    Euclid_prof = galsim.Convolve(gal, Euclid_PSF)

    print "Drawing HST images"
    # Draw HST images
    HST_images = [HST_prof.drawImage(rband, nx=64, ny=64, scale=0.05),
                  HST_prof.drawImage(iband, nx=64, ny=64, scale=0.05)]

    print "Drawing Euclid image"
    Euclid_image = Euclid_prof.drawImage(visband, nx=32, ny=32, scale=0.1)

    # Now "deconvolve" the chromatic HST PSF while asserting the correct SEDs.
    print "Constructing ChromaticRealGalaxy"
    crg = galsim.ChromaticRealGalaxy((HST_images,
                                      [rband, iband],
                                      [const_SED, linear_SED],
                                      HST_PSF))

    # crg should be effectively the same thing as gal now.  Let's test.

    Euclid_recon_image = (galsim.Convolve(crg, Euclid_PSF)
                          .drawImage(visband, nx=32, ny=32, scale=0.1))

    np.testing.assert_almost_equal(Euclid_image.array.max()/Euclid_recon_image.array.max(),
                                   1.0, 2)
    np.testing.assert_almost_equal(Euclid_image.array.sum()/Euclid_recon_image.array.sum(),
                                   1.0, 2)
    print "Max comparison:", Euclid_image.array.max(), Euclid_recon_image.array.max()
    print "Sum comparison:", Euclid_image.array.sum(), Euclid_recon_image.array.sum()
    if False:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13,10))
        ax = fig.add_subplot(231)
        im = ax.imshow(HST_images[0].array)
        plt.colorbar(im)
        ax.set_title('rband')
        ax = fig.add_subplot(232)
        im = ax.imshow(HST_images[1].array)
        plt.colorbar(im)
        ax.set_title('iband')
        ax = fig.add_subplot(234)
        im = ax.imshow(Euclid_image.array)
        plt.colorbar(im)
        ax.set_title('Euclid')
        ax = fig.add_subplot(235)
        im = ax.imshow(Euclid_recon_image.array)
        plt.colorbar(im)
        ax.set_title('Euclid reconstruction')
        ax = fig.add_subplot(236)
        im = ax.imshow(Euclid_recon_image.array - Euclid_image.array, cmap='seismic',
                       vmin=-0.005, vmax=0.005)
        plt.colorbar(im)
        ax.set_title('Euclid residual')
        plt.tight_layout()
        plt.show()
    np.testing.assert_array_almost_equal(Euclid_image.array/Euclid_image.array.max(),
                                         Euclid_recon_image.array/Euclid_image.array.max(),
                                         3) # Fails at 4th decimal

    # Other tests:
    #     - draw the same image as origin?
    #     - compare intermediate products: does aj match the input spatial profiles?
    #       (are there degeneracies?)
    #     - stupid tests like using only one filter should perform similarly to RealGalaxy?
    #     - ellipticity tests like those above for RealGalaxy?


if __name__ == "__main__":
    test_real_galaxy_ideal()
    test_real_galaxy_saved()
    test_chromatic_real_galaxy()
