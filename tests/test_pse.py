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
def test_PSE_basic():
    """Basic test of power spectrum estimation.
    """

    # Here are some parameters that define array sizes and other such things.
    array_size = 300
    e_tolerance = 0.10     # 10% error allowed because of finite grid effects, noise fluctuations,
                           # and other things.  This unit test is just for a basic sanity test.
    b_tolerance = 0.15     # B-mode is slightly less accurate.
    zero_tolerance = 0.03  # For power that should be zero

    n_ell = 8
    grid_spacing = 0.1 # degrees
    ps_file = os.path.join(datapath, 'cosmo-fid.zmed1.00.out')
    rand_seed = 2718

    # Begin by setting up the PowerSpectrum and generating shears.
    tab = galsim.LookupTable.from_file(ps_file)
    ps = galsim.PowerSpectrum(tab, units=galsim.radians)
    g1, g2 = ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                          rng=galsim.BaseDeviate(rand_seed))

    # Then initialize the PSE object.
    pse = galsim.pse.PowerSpectrumEstimator(N=array_size,
                                            sky_size_deg=array_size*grid_spacing,
                                            nbin=n_ell)

    check_pickle(pse)

    # Estimate the power spectrum using the PSE, without weighting.
    ell, P_e1, P_b1, P_eb1 = pse.estimate(g1, g2)

    # To check: P_E is right (to within the desired tolerance); P_B and P_EB are <1% of P_E.
    P_theory = np.zeros_like(ell)
    for ind in range(len(ell)):
        P_theory[ind] = tab(ell[ind])
    # Note: we don't check the first element because at low ell the tests can fail more
    # spectacularly for reasons that are well understood.
    np.testing.assert_allclose(P_e1[1:], P_theory[1:], rtol=e_tolerance,
                               err_msg='PSE returned wrong E power')
    np.testing.assert_allclose(P_b1[1:]/P_theory[1:], 0., atol=zero_tolerance,
                               err_msg='PSE found B power')
    np.testing.assert_allclose(P_eb1[1:]/P_theory[1:], 0., atol=zero_tolerance,
                               err_msg='PSE found EB cross-power')

    # Test theory_func
    ell, P_e1, P_b1, P_eb1, t = pse.estimate(g1, g2, theory_func=tab)
    # This isn't super accurate.  I think just because of binning.  But I'm not sure.
    np.testing.assert_allclose(t, P_theory, rtol=0.3,
                               err_msg='PSE returned wrong theory binning')

    # Also check the case where P_e=P_b.
    ps = galsim.PowerSpectrum(tab, tab, units=galsim.radians)
    g1, g2 = ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                          rng=galsim.BaseDeviate(rand_seed))
    ell, P_e2, P_b2, P_eb2 = pse.estimate(g1, g2)
    np.testing.assert_allclose(P_e2[1:], P_theory[1:], rtol=e_tolerance,
                               err_msg='PSE returned wrong E power')
    np.testing.assert_allclose(P_b2[1:], P_theory[1:], rtol=b_tolerance,
                               err_msg='PSE returned wrong B power')
    np.testing.assert_allclose(P_eb2[1:]/P_theory[1:], 0., atol=zero_tolerance,
                               err_msg='PSE found EB cross-power')

    # And check the case where P_b is nonzero and P_e is zero.
    ps = galsim.PowerSpectrum(e_power_function=None, b_power_function=tab,
                              units=galsim.radians)
    g1, g2 = ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                          rng=galsim.BaseDeviate(rand_seed))
    ell, P_e3, P_b3, P_eb3 = pse.estimate(g1, g2)
    np.testing.assert_allclose(P_e3[1:]/P_theory[1:], 0., atol=zero_tolerance,
                               err_msg='PSE found E power when it should be zero')
    np.testing.assert_allclose(P_b3[1:], P_theory[1:], rtol=b_tolerance,
                               err_msg='PSE returned wrong B power')
    np.testing.assert_allclose(P_eb3[1:]/P_theory[1:], 0., atol=zero_tolerance,
                               err_msg='PSE found EB cross-power')

    assert_raises(ValueError, pse.estimate, g1[:3,:3], g2)
    assert_raises(ValueError, pse.estimate, g1[:3,:8], g2[:3,:8])
    assert_raises(ValueError, pse.estimate, g1[:8,:8], g2[:8,:8])


@timer
def test_PSE_weight():
    """Test of power spectrum estimation with weights.
    """
    array_size = 300
    n_ell = 8
    grid_spacing = 0.1
    ps_file = os.path.join(datapath, 'cosmo-fid.zmed1.00.out')
    rand_seed = 2718

    tab = galsim.LookupTable.from_file(ps_file)
    ps = galsim.PowerSpectrum(tab, units=galsim.radians)
    g1, g2 = ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                          rng=galsim.BaseDeviate(rand_seed))

    pse = galsim.pse.PowerSpectrumEstimator(N=array_size,
                                            sky_size_deg=array_size*grid_spacing,
                                            nbin=n_ell)

    ell, P_e1, P_b1, P_eb1, P_theory = pse.estimate(g1, g2, weight_EE=True, weight_BB=True,
                                                    theory_func=tab)
    print('P_e1 = ',P_e1)
    print('rel_diff = ',(P_e1-P_theory)/P_theory)
    print('rel_diff using P[1] = ',(P_e1-P_theory)/P_theory[1])
    # The agreement here seems really bad.  Should I not expect these to be closer than this?
    eb_tolerance = 0.4
    zero_tolerance = 0.03

    np.testing.assert_allclose(P_e1[1:], P_theory[1:], rtol=eb_tolerance,
                               err_msg='Weighted PSE returned wrong E power')

    np.testing.assert_allclose(P_b1/P_theory, 0., atol=zero_tolerance,
                               err_msg='Weighted PSE found B power')
    print(P_eb1/P_theory)
    np.testing.assert_allclose(P_eb1/P_theory, 0., atol=zero_tolerance,
                               err_msg='Weighted PSE found EB cross-power')

    # Also check the case where P_e=P_b.
    ps = galsim.PowerSpectrum(tab, tab, units=galsim.radians)
    g1, g2 = ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                          rng=galsim.BaseDeviate(rand_seed))
    ell, P_e2, P_b2, P_eb2 = pse.estimate(g1, g2, weight_EE=True, weight_BB=True)
    np.testing.assert_allclose(P_e2[1:], P_theory[1:], rtol=eb_tolerance,
                               err_msg='Weighted PSE returned wrong E power')
    np.testing.assert_allclose(P_b2[1:], P_theory[1:], rtol=eb_tolerance,
                               err_msg='Weighted PSE returned wrong B power')
    np.testing.assert_allclose(P_eb2[1:]/P_theory[1:], 0., atol=zero_tolerance,
                               err_msg='Weighted PSE found EB cross-power')

    # And check the case where P_b is nonzero and P_e is zero.
    ps = galsim.PowerSpectrum(e_power_function=None, b_power_function=tab,
                              units=galsim.radians)
    g1, g2 = ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                          rng=galsim.BaseDeviate(rand_seed))
    ell, P_e3, P_b3, P_eb3 = pse.estimate(g1, g2, weight_EE=True, weight_BB=True)
    np.testing.assert_allclose(P_e3[1:]/P_theory[1:], 0., atol=zero_tolerance,
                               err_msg='Weighted PSE found E power when it should be zero')
    np.testing.assert_allclose(P_b3[1:], P_theory[1:], rtol=eb_tolerance,
                               err_msg='Weighted PSE returned wrong B power')
    np.testing.assert_allclose(P_eb3[1:]/P_theory[1:], 0., atol=zero_tolerance,
                               err_msg='Weighted PSE found EB cross-power')

    assert_raises(TypeError, pse.estimate, g1, g2, weight_EE=8)
    assert_raises(TypeError, pse.estimate, g1, g2, weight_BB='yes')

    # If N is fairly small, then can get zeros in the counts, which raises an error
    array_size = 5
    g1, g2 = ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                          rng=galsim.BaseDeviate(rand_seed))
    pse = galsim.pse.PowerSpectrumEstimator(N=array_size,
                                            sky_size_deg=array_size*grid_spacing,
                                            nbin=n_ell)
    with assert_raises(galsim.GalSimError):
        pse.estimate(g1,g2)



if __name__ == "__main__":
    runtests(__file__)
