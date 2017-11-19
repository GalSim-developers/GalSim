# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
import time

from galsim_test_helpers import *

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../examples/data/"))

# Here are some parameters that define array sizes and other such things.
array_size = 300
tolerance = 0.05  # 10% error allowed because of finite grid effects, noise fluctuations, and other
                  # things.  This unit test is just for a basic sanity test.
zero_tolerance = 0.01 # For power that should be zero, allow it to be <0.02 * the nonzero
                      # ones.
n_ell = 8
grid_spacing = 0.1 # degrees
ps_file = os.path.join(datapath, 'cosmo-fid.zmed1.00.out')
rand_seed = 2718


@timer
def test_PSE_basic():
    """Basic test of power spectrum estimation.
    """
    # Begin by setting up the PowerSpectrum and generating shears.
    my_tab = galsim.LookupTable.from_file(ps_file)
    my_ps = galsim.PowerSpectrum(my_tab, units=galsim.radians)
    g1, g2 = my_ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                             rng=galsim.BaseDeviate(rand_seed))

    # Then initialize the PSE object.
    my_pse = galsim.pse.PowerSpectrumEstimator(N=array_size,
                                               sky_size_deg=array_size*grid_spacing,
                                               nbin=n_ell)

    do_pickle(my_pse)

    # Estimate the power spectrum using the PSE, without weighting.
    ell, P_e1, P_b1, P_eb1 = my_pse.estimate(g1, g2)

    # To check: P_E is right (to within the desired tolerance); P_B and P_EB are <1% of P_E.
    P_e_theory = np.zeros_like(ell)
    for ind in range(len(ell)):
        P_e_theory[ind] = my_tab(ell[ind])
    # Note: we don't check the first element because at low ell the tests can fail more
    # spectacularly for reasons that are well understood.
    np.testing.assert_array_almost_equal(
        (P_e1[1:]/P_e_theory[1:]-1.)/(2*tolerance), 0., decimal=0,
        err_msg='PSE returned wrong E power')
    np.testing.assert_array_almost_equal(
        (P_b1[1:]/P_e_theory[1:])/(2*zero_tolerance), 0., decimal=0,
         err_msg='PSE found B power')
    np.testing.assert_array_almost_equal(
        (P_eb1[1:]/P_e_theory[1:])/(2*zero_tolerance), 0., decimal=0,
         err_msg='PSE found EB cross-power')

    # Also check the case where P_e=P_b.
    my_ps = galsim.PowerSpectrum(my_tab, my_tab, units=galsim.radians)
    g1, g2 = my_ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                             rng=galsim.BaseDeviate(rand_seed))
    ell, P_e2, P_b2, P_eb2 = my_pse.estimate(g1, g2)
    np.testing.assert_array_almost_equal(
        (P_e2[1:]/P_e_theory[1:]-1.)/(2*tolerance), 0., decimal=0,
        err_msg='PSE returned wrong E power')
    np.testing.assert_array_almost_equal(
        (P_b2[1:]/P_e_theory[1:]-1.)/(2*tolerance), 0., decimal=0,
        err_msg='PSE returned wrong B power')
    np.testing.assert_array_almost_equal(
        (P_eb2[1:]/P_e_theory[1:])/(2*zero_tolerance), 0., decimal=0,
        err_msg='PSE found EB cross-power')

    # And check the case where P_b is nonzero and P_e is zero.
    my_ps = galsim.PowerSpectrum(e_power_function=None, b_power_function=my_tab,
                                 units=galsim.radians)
    g1, g2 = my_ps.buildGrid(grid_spacing=grid_spacing, ngrid=array_size, units=galsim.degrees,
                             rng=galsim.BaseDeviate(rand_seed))
    ell, P_e3, P_b3, P_eb3 = my_pse.estimate(g1, g2)
    np.testing.assert_array_almost_equal(
        (P_e3[1:]/P_e_theory[1:])/(2*zero_tolerance), 0., decimal=0,
        err_msg='PSE found E power when it should be zero')
    np.testing.assert_array_almost_equal(
        (P_b3[1:]/P_e_theory[1:]-1.)/(2*tolerance), 0., decimal=0,
        err_msg='PSE returned wrong B power')
    np.testing.assert_array_almost_equal(
        (P_eb3[1:]/P_e_theory[1:])/(2*zero_tolerance), 0., decimal=0,
        err_msg='PSE found EB cross-power')

if __name__ == "__main__":
    test_PSE_basic()
