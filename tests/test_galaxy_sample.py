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

import os
import numpy as np

import galsim
from galsim_test_helpers import *


path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))


@timer
def test_cosmos_basic():
    """Check some basic functionality of the COSMOSCatalog class."""
    # Note, there's not much here yet.   Could try to think of other tests that are more
    # interesting.

    # Initialize a COSMOSCatalog with all defaults.
    cat = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                               dir=datapath)

    # Check GalaxySample equivalent
    cat1 = galsim.GalaxySample('real_galaxy_catalog_23.5_example.fits', datapath,
                               cut_ratio=0.2, sn_limit=20.)
    assert cat1.nobjects == cat1.getNObjects() == cat.nobjects == cat.getNObjects()
    assert cat1 == cat  # These (intentionally) test as equal even though different classes.

    # Initialize one that doesn't exclude failures.  It should be >= the previous one in length.
    cat2 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                               dir=datapath, exclusion_level='none')
    assert cat2.nobjects >= cat.nobjects
    assert len(cat2) == cat2.nobjects == cat2.getNTot() == 100
    assert len(cat) == cat.nobjects < cat.getNTot()

    # Check other exclusion levels:
    cat3 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='bad_stamp')
    assert len(cat3) == 97
    cat4 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='bad_fits')
    assert len(cat4) == 100  # no bad fits in the example file as it happens.
    cat5 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='marginal')
    assert len(cat5) == 96   # this is actually the default, so == cat
    assert cat == cat5
    cat6 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='marginal',
                                max_hlr=2, max_flux=2000)
    assert len(cat6) == 93
    cat7 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='marginal',
                                min_hlr=0.2, min_flux=10)
    assert len(cat7) == 91

    # Check the 25.2 exclusions.  We don't have a 25.2 catalog available in Travis runs, so
    # mock up the example catalog as though it were 25.2
    # Also check the min/max hlr and flux options.
    cat2.use_sample = '25.2'
    hlr = cat2.param_cat['hlr'][:,0]
    flux = cat2.param_cat['flux'][:,0]
    print("Full range of hlr = ", np.min(hlr), np.max(hlr))
    print("Full range of flux = ", np.min(flux), np.max(flux))
    cat2._apply_exclusion('marginal', min_hlr=0.2, max_hlr=2, min_flux=50, max_flux=5000)
    print("Size of catalog with hlr and flux exclusions == ",len(cat2))
    assert len(cat2) == 47

    # Repeat with a fake 25.2 catalog (just symlinks to the above 23.5 catalog).
    fake_25_2_cat = galsim.COSMOSCatalog(file_name='fake_25.2.fits', dir='input',
                                         exclusion_level='marginal', min_hlr=0.2, max_hlr=2,
                                         min_flux=50, max_flux=5000)
    assert len(fake_25_2_cat) == 47

    # Check for reasonable exceptions when initializing.
    # Can't find data (wrong directory).
    with assert_raises(OSError):
        galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits')

    # Try making galaxies
    gal_real = cat.makeGalaxy(index=0,gal_type='real',chromatic=False)
    assert isinstance(gal_real, galsim.RealGalaxy)

    gal_param = cat.makeGalaxy(index=10,gal_type='parametric',chromatic=True)
    assert isinstance(gal_param, galsim.ChromaticObject)

    # Second time through, don't make the bandpass.
    bp = cat._bandpass
    sed = cat._sed
    assert bp is not None
    gal_param2 = cat.makeGalaxy(index=13, gal_type='parametric', chromatic=True)
    assert isinstance(gal_param2, galsim.ChromaticObject)
    assert gal_param != gal_param2
    assert cat._bandpass is bp   # Not just ==.  "is" means the same object.
    assert cat._sed is sed

    # So far, we've made a bulge+disk and a disky Sersic.
    # Do two more to run through two more paths in the code.
    gal_param3 = cat.makeGalaxy(index=50, gal_type='parametric', chromatic=True)
    gal_param4 = cat.makeGalaxy(index=67, gal_type='parametric', chromatic=True)

    gal_real_list = cat.makeGalaxy(index=[3,6],gal_type='real',chromatic=False)
    for gal_real in gal_real_list:
        assert isinstance(gal_real, galsim.RealGalaxy)

    gal_param_list = cat.makeGalaxy(index=[4,7],gal_type='parametric',chromatic=False)
    for gal_param in gal_param_list:
        assert isinstance(gal_param, galsim.GSObject)

    # Check for parametric catalog
    # Can give either the regular name or the _fits name.
    cat_param = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                     dir=datapath, use_real=False)
    cat_param2 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example_fits.fits',
                                      dir=datapath, use_real=False)
    assert cat_param2 == cat_param

    # Try making galaxies
    gal = cat_param.makeGalaxy(index=1)
    assert isinstance(gal, galsim.GSObject)

    gal_list = cat_param.makeGalaxy(index=[2,3])
    for gal in gal_list:
        assert isinstance(gal, galsim.GSObject)

    # Check sersic_prec option.
    sersic0 = cat_param.makeGalaxy(index=59, sersic_prec=0)
    np.testing.assert_almost_equal(sersic0.original.n, 1.14494567108)
    sersic1 = cat_param.makeGalaxy(index=59, sersic_prec=0.05)  # The default.
    np.testing.assert_almost_equal(sersic1.original.n, 1.15)
    sersic2 = cat_param.makeGalaxy(index=59, sersic_prec=0.1)
    np.testing.assert_almost_equal(sersic2.original.n, 1.1)
    sersic3 = cat_param.makeGalaxy(index=59, sersic_prec=0.5)
    np.testing.assert_almost_equal(sersic3.original.n, 1.0)

    assert_raises(TypeError, galsim.COSMOSCatalog, 'real_galaxy_catalog_23.5_example.fits',
                  dir=datapath, sample='23.5')
    assert_raises(ValueError, galsim.COSMOSCatalog, sample='invalid')

    assert_raises(ValueError, cat_param.makeGalaxy, gal_type='real')
    assert_raises(ValueError, cat_param.makeGalaxy, gal_type='invalid')
    assert_raises(ValueError, cat.makeGalaxy, gal_type='invalid')
    assert_raises(TypeError, cat_param.makeGalaxy, rng='invalid')

    assert_raises(NotImplementedError, cat.makeGalaxy, gal_type='real', chromatic=True)

@timer
def test_cosmos_fluxnorm():
    """Check for flux normalization properties of COSMOSCatalog class."""
    # Check that if we make a RealGalaxy catalog, and a COSMOSCatalog, and draw the real object, the
    # fluxes should match very well.  These correspond to 1s exposures.
    test_ind = 54
    rand_seed = 12345
    cat = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                               dir=datapath, exclusion_level='none')
    rgc = galsim.RealGalaxyCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                   dir=datapath)
    final_psf = galsim.Airy(diam=1.2, lam=800.) # PSF twice as big as HST in F814W.
    gal1 = cat.makeGalaxy(test_ind, gal_type='real', rng=galsim.BaseDeviate(rand_seed))
    gal2 = galsim.RealGalaxy(rgc, index=test_ind, rng=galsim.BaseDeviate(rand_seed))
    gal1 = galsim.Convolve(gal1, final_psf)
    gal2 = galsim.Convolve(gal2, final_psf)
    im1 = gal1.drawImage(scale=0.05)
    im2 = gal2.drawImage(scale=0.05)

    # Then check that if we draw a parametric representation that is achromatic, that the flux
    # matches reasonably well (won't be exact because model isn't perfect).
    gal1_param = cat.makeGalaxy(test_ind, gal_type='parametric', chromatic=False)
    gal1_param_final = galsim.Convolve(gal1_param, final_psf)
    im1_param = gal1_param_final.drawImage(scale=0.05)

    # Then check the same for a chromatic parametric representation that is drawn into the same
    # band.
    bp_file = os.path.join(galsim.meta_data.share_dir, 'bandpasses', 'ACS_wfc_F814W.dat')
    bandpass = galsim.Bandpass(bp_file, wave_type='nm').withZeropoint(25.94)#34.19)
    gal1_chrom = cat.makeGalaxy(test_ind, gal_type='parametric', chromatic=True)
    gal1_chrom = galsim.Convolve(gal1_chrom, final_psf)
    im1_chrom = gal1_chrom.drawImage(bandpass, scale=0.05)

    ref_val = [im1.array.sum(), im1.array.sum(), im1.array.sum()]
    test_val = [im2.array.sum(), im1_param.array.sum(), im1_chrom.array.sum()]
    np.testing.assert_allclose(ref_val, test_val, rtol=0.1,
                               err_msg='Flux normalization problem in COSMOS galaxies')

    # Finally, check that the original COSMOS info is stored properly after transformations, for
    # both Sersic galaxies (like galaxy 0 in the catalog) and the one that is gal1_param above.
    gal0_param = cat.makeGalaxy(0, gal_type='parametric', chromatic=False)
    assert hasattr(gal0_param.shear(g1=0.05).original, 'index'), \
        'Sersic galaxy does not retain index information after transformation'
    assert hasattr(gal1_param.shear(g1=0.05).original, 'index'), \
        'Bulge+disk galaxy does not retain index information after transformation'

    assert_raises(ValueError, galsim.COSMOSCatalog, 'real_galaxy_catalog_23.5_example.fits',
                  dir=datapath, exclusion_level='invalid')

    # Check scaling with area and exptime
    print('hst area = ',galsim.COSMOSCatalog.hst_eff_area)
    assert np.isclose(galsim.COSMOSCatalog.hst_eff_area,
                      np.pi * 2.4**2 / 4 * (1-0.33**2) * 100**2)
    cat2 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                dir=datapath, exclusion_level='none', area=3456, exptime=2.3)
    flux_scaling = 2.3 * (3456/galsim.COSMOSCatalog.hst_eff_area)
    gal1 = cat.makeGalaxy(test_ind, gal_type='parametric')
    gal2 = cat2.makeGalaxy(test_ind, gal_type='parametric')
    assert np.isclose(gal2.flux, gal1.flux * flux_scaling)

@timer
def test_cosmos_random():
    """Check the random object functionality of the COSMOS catalog."""
    # For most of this test, we'll use the selectRandomIndex() routine, which does not try to
    # construct the GSObjects.  This makes the test go fast.  However, we will at the end have a
    # test to ensure that calling makeGalaxy() while requesting a random object has the same
    # behavior as using selectRandomIndex() in limited cases.

    # Initialize the catalog.  The first will have weights, while the second will not (since they
    # are in the RealGalaxyCatalog).
    cat = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                               dir=datapath)
    cat_param = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                     dir=datapath, use_real=False)
    assert cat_param.real_cat is None
    assert cat.real_cat is not None

    # Check for exception handling if bad inputs given for the random functionality.
    assert_raises(ValueError, cat.selectRandomIndex, 0)
    assert_raises(ValueError, cat.selectRandomIndex, 10.7)
    assert_raises(TypeError, cat.selectRandomIndex, 10, rng=3)

    # Check that random objects give the right <weight> without/with weighting.
    wt = cat.real_cat.weight[cat.orig_index]
    wt /= np.max(wt)
    avg_weight_val = np.sum(wt)/len(wt)
    wavg_weight_val = np.sum(wt**2)/np.sum(wt)
    with assert_raises(AssertionError):
        np.testing.assert_almost_equal(avg_weight_val, wavg_weight_val, 3)
    # Make sure we use enough objects that the mean weights converge properly.
    randind_wt = cat.selectRandomIndex(30000, rng=galsim.BaseDeviate(1234))
    wtrand = cat.real_cat.weight[cat.orig_index[randind_wt]] / \
        np.max(cat.real_cat.weight[cat.orig_index])
    # The average value of wtrand should be wavgw_weight_val in this case, since we used the weights
    # to probabilistically select galaxies.
    np.testing.assert_almost_equal(np.mean(wtrand), wavg_weight_val,3,
                                   err_msg='Average weight for random sample is wrong')

    # The param-only catalog doesn't have weights, so it does unweighted selection, which emits a
    # warning.  We know about this and want to ignore it here.
    with assert_warns(galsim.GalSimWarning):
        randind = cat_param.selectRandomIndex(30000, rng=galsim.BaseDeviate(1234))
    wtrand = cat.real_cat.weight[cat.orig_index[randind]] / \
        np.max(cat.real_cat.weight[cat.orig_index])
    # The average value of wtrand should be avg_weight_val, since we did not do a weighted
    # selection.
    np.testing.assert_almost_equal(np.mean(wtrand), avg_weight_val,3,
                                   err_msg='Average weight for random sample is wrong')

    # Check for consistency of randoms with same random seed.  Do this both for the weighted and the
    # unweighted calculation.
    # Check for inconsistency of randoms with different random seed, or same seed but without/with
    # weighting.
    rng1 = galsim.BaseDeviate(1234)
    rng2 = galsim.BaseDeviate(1234)
    ind1 = cat.selectRandomIndex(10, rng=rng1)
    ind2 = cat.selectRandomIndex(10, rng=rng2)
    np.testing.assert_array_equal(ind1,ind2,
                                  err_msg='Different random indices selected with same seed!')
    with assert_warns(galsim.GalSimWarning):
        ind1p = cat_param.selectRandomIndex(10, rng=rng1)
    with assert_warns(galsim.GalSimWarning):
        ind2p = cat_param.selectRandomIndex(10, rng=rng2)
    np.testing.assert_array_equal(ind1p,ind2p,
                                  err_msg='Different random indices selected with same seed!')
    rng3 = galsim.BaseDeviate(5678)
    ind3 = cat.selectRandomIndex(10, rng=rng3)
    ind3p = cat.selectRandomIndex(10) # initialize RNG based on time
    assert_raises(AssertionError, np.testing.assert_array_equal, ind1, ind1p)
    assert_raises(AssertionError, np.testing.assert_array_equal, ind1, ind3)
    assert_raises(AssertionError, np.testing.assert_array_equal, ind1p, ind3p)

    # Finally, make sure that directly calling selectRandomIndex() gives the same random ones as
    # makeGalaxy().  We'll do one real object because they are slower, and multiple parametric (just
    # to make sure that the multi-object selection works consistently).
    use_seed = 567
    obj = cat.makeGalaxy(rng=galsim.BaseDeviate(use_seed))
    ind = cat.selectRandomIndex(1, rng=galsim.BaseDeviate(use_seed))
    obj_2 = cat.makeGalaxy(ind)
    # Note: for real galaxies we cannot require that obj==obj_2, just that obj.index==obj_2.index.
    # That's because we want to make sure the same galaxy is being randomly selected, but we cannot
    # require that noise padding be the same, given the inconsistency in how the BaseDeviates are
    # used in the above cases.
    assert obj.index==obj_2.index,'makeGalaxy selects random objects inconsistently'

    n_random = 3
    objs = cat.makeGalaxy(rng=galsim.BaseDeviate(use_seed), gal_type='parametric',
                          n_random=n_random)
    inds = cat.selectRandomIndex(n_random, rng=galsim.BaseDeviate(use_seed))
    objs_2 = cat_param.makeGalaxy(inds)
    for i in range(n_random):
        # With parametric objects there is no noise padding, so we can require completely identical
        # GSObjects (not just equal indices).
        assert objs[i]==objs_2[i],'makeGalaxy selects random objects inconsistently'

    # Finally, check for consistency with random object selected from RealGalaxyCatalog.  For this
    # case, we need to make another COSMOSCatalog that does not flag the bad objects.
    use_seed=31415
    cat = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                               dir=datapath, exclusion_level='none')
    rgc = galsim.RealGalaxyCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                                   dir=datapath)
    ind_cc = cat.selectRandomIndex(1, rng=galsim.BaseDeviate(use_seed))
    foo = galsim.RealGalaxy(rgc, random=True, rng=galsim.BaseDeviate(use_seed))
    ind_rgc = foo.index
    assert ind_cc==ind_rgc,\
        'Different weighted random index selected from COSMOSCatalog and RealGalaxyCatalog'

    # Also check for the unweighted case.  Just remove that info from the catalogs and redo the
    # test.
    cat.real_cat = None
    rgc.weight = None
    with assert_warns(galsim.GalSimWarning):
        ind_cc = cat.selectRandomIndex(1, rng=galsim.BaseDeviate(use_seed))
    foo = galsim.RealGalaxy(rgc, random=True, rng=galsim.BaseDeviate(use_seed))
    ind_rgc = foo.index
    assert ind_cc==ind_rgc,\
        'Different unweighted random index selected from COSMOSCatalog and RealGalaxyCatalog'

    # Check that setting _n_rng_calls properly tracks the RNG calls for n_random=1 and >1.
    cat = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                               dir=datapath, exclusion_level='none')
    test_seed = 123456
    ud = galsim.UniformDeviate(test_seed)
    obj, n_rng_calls = cat.selectRandomIndex(1, rng=ud, _n_rng_calls=True)
    ud2 = galsim.UniformDeviate(test_seed)
    ud2.discard(n_rng_calls)
    assert ud()==ud2(), '_n_rng_calls kwarg did not give proper tracking of RNG calls'
    ud = galsim.UniformDeviate(test_seed)
    obj, n_rng_calls = cat.selectRandomIndex(17, rng=ud, _n_rng_calls=True)
    ud2 = galsim.UniformDeviate(test_seed)
    ud2.discard(n_rng_calls)
    assert ud()==ud2(), '_n_rng_calls kwarg did not give proper tracking of RNG calls'

    # Invalid to both privide index and ask for random selection
    with assert_raises(galsim.GalSimIncompatibleValuesError):
        cat_param.makeGalaxy(index=(11,13,17), n_random=3)

@timer
def test_cosmos_deep():
    """Test the deep option of makeGalaxy"""

    cat = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits', dir=datapath)

    # Pick a random galaxy
    # Turn off noise padding to make the comparisons more reliable.
    gal_shallow = cat.makeGalaxy(index=17, gal_type='real', noise_pad_size=0)
    print('gal_shallow = ',gal_shallow)
    shallow_flux = gal_shallow.flux
    shallow_hlr = gal_shallow.calculateHLR()
    print('flux = ',shallow_flux)
    print('hlr = ',shallow_hlr)

    gal_deep = cat.makeGalaxy(index=17, gal_type='real', deep=True, noise_pad_size=0)
    print('gal_deep = ',gal_deep)
    deep_flux = gal_deep.flux
    deep_hlr = gal_deep.calculateHLR()
    print('flux = ',deep_flux)
    print('hlr = ',deep_hlr)

    # Deep galaxy is fainter and smaller.
    np.testing.assert_almost_equal(deep_flux / shallow_flux, 10.**(-0.6))
    np.testing.assert_almost_equal(deep_hlr / shallow_hlr, 0.6)

    # With samples other than 23.5, it raises a warning and doesn't do any scaling.
    cat.use_sample = '25.2'
    with assert_warns(galsim.GalSimWarning):
        gal_not_deep = cat.makeGalaxy(index=17, gal_type='real', deep=True, noise_pad_size=0)
    assert gal_not_deep.flux == shallow_flux
    assert gal_not_deep.calculateHLR() == shallow_hlr

    cat.use_sample = 'user_defined'
    with assert_warns(galsim.GalSimWarning):
        gal_not_deep = cat.makeGalaxy(index=17, gal_type='real', deep=True, noise_pad_size=0)
    assert gal_not_deep.flux == shallow_flux
    assert gal_not_deep.calculateHLR() == shallow_hlr


if __name__ == "__main__":
    runtests(__file__)
