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
import os
import numpy as np
import sys

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
    # Initialize one that doesn't exclude failures.  It should be >= the previous one in length.
    cat2 = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits',
                               dir=datapath, exclusion_level='none')
    assert cat2.nobjects>=cat.nobjects

    # Check for reasonable exceptions when initializing.
    # Can't find data (wrong directory).
    with assert_raises(IOError):
        galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example.fits')

    # Try making galaxies
    gal_real = cat2.makeGalaxy(index=0,gal_type='real',chromatic=False)
    if not isinstance(gal_real, galsim.RealGalaxy):
        raise TypeError("COSMOS Catalog makeGalaxy routine does not return an instance of "
                        "'galsim.RealGalaxy'")

    gal_param = cat.makeGalaxy(index=10,gal_type='parametric',chromatic=True)
    if not isinstance(gal_param, galsim.ChromaticObject):
        raise TypeError("COSMOS Catalog makeGalaxy routine does not return an instance of "
                        "'galsim.ChromaticObject' for parametric galaxies")

    gal_real_list = cat.makeGalaxy(index=[3,6],gal_type='real',chromatic=False)
    for gal_real in gal_real_list:
        if not isinstance(gal_real, galsim.RealGalaxy):
            raise TypeError("COSMOS Catalog makeGalaxy routine does not return a list of instances "
                            "of 'galsim.RealGalaxy'")

    gal_param_list = cat.makeGalaxy(index=[4,7],gal_type='parametric',chromatic=False)
    for gal_param in gal_param_list:
        if not isinstance(gal_param, galsim.GSObject):
            raise TypeError("COSMOS Catalog makeGalaxy routine does not return a list of instances "
                            "of 'galsim.GSObect'")

    # Check for parametric catalog
    cat_param = galsim.COSMOSCatalog(file_name='real_galaxy_catalog_23.5_example_fits.fits',
                                     dir=datapath, use_real=False)

    # Try making galaxies
    gal = cat_param.makeGalaxy(index=1)
    if not isinstance(gal, galsim.GSObject):
        raise TypeError("COSMOS Catalog makeGalaxy routine does not return an instance of "
                        "'galsim.GSObject when loaded from a fits file.")

    gal_list = cat_param.makeGalaxy(index=[2,3])
    for gal in gal_list:
        if not isinstance(gal, galsim.GSObject):
            raise TypeError("COSMOS Catalog makeGalaxy routine does not return a list of instances "
                            "of 'galsim.GSObject when loaded from a fits file.")


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
    assert hasattr(cat, 'real_cat')
    assert not hasattr(cat_param, 'real_cat')

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
    # From here on we need to suppress some warnings that come from calling cat_param.  It doesn't
    # have weights, so it does unweighted selection, which emits a warning.  We know about this and
    # don't want to spit out the warning each time, so suppress it.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        ind1p = cat_param.selectRandomIndex(10, rng=rng1)
        ind2p = cat_param.selectRandomIndex(10, rng=rng2)
        np.testing.assert_array_equal(ind1p,ind2p,
                                      err_msg='Different random indices selected with same seed!')
        rng3 = galsim.BaseDeviate(5678)
        ind3 = cat.selectRandomIndex(10, rng=rng3)
        ind3p = cat_param.selectRandomIndex(10) # initialize RNG based on time
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
        objs = cat_param.makeGalaxy(rng=galsim.BaseDeviate(use_seed), n_random=n_random)
        inds = cat_param.selectRandomIndex(n_random, rng=galsim.BaseDeviate(use_seed))
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
        del cat.real_cat
        del rgc.weight
        ind_cc = cat.selectRandomIndex(1, rng=galsim.BaseDeviate(use_seed))
        foo = galsim.RealGalaxy(rgc, random=True, rng=galsim.BaseDeviate(use_seed))
        ind_rgc = foo.index
        assert ind_cc==ind_rgc,\
            'Different unweighted random index selected from COSMOSCatalog and RealGalaxyCatalog'

        # Check that setting _n_rng_calls properly tracks the RNG calls for n_random=1 and >1.
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

if __name__ == "__main__":
    test_cosmos_basic()
    test_cosmos_fluxnorm()
    test_cosmos_random()
