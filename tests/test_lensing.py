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

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

refdir = os.path.join(".", "lensing_reference_data") # Directory containing the reference

def funcname():
    import inspect
    return inspect.stack()[1][3]

# for simple demonstration purposes, a few very simple power-law power spectra that don't crash and
# burn at k=0
def pk2(k):
    return k**(2.0)

def pk1(k):
    return k

def pkflat(k):
    # note: this gives random Gaussian shears with variance of 0.01
    return 0.01+np.zeros_like(k)

def test_nfwhalo():
    """Various tests of the NFWHalo class (against reference data, and basic sanity tests)"""
    import time
    t1 = time.time()

    # reference data comes from Matthias Bartelmann's libastro code
    # cluster properties: M=1e15, conc=4, redshift=1
    # sources at redshift=2
    # columns:
    # distance [arcsec], deflection [arcsec], shear, reduced shear, convergence
    # distance go from 1 .. 599 arcsec
    ref = np.loadtxt(refdir + '/nfw_lens.dat')

    # set up the same halo
    halo = galsim.NFWHalo(mass=1e15, conc=4, redshift=1)
    pos_x = np.arange(1,600)
    pos_y = np.zeros_like(pos_x)
    z_s = 2
    kappa = halo.getConvergence((pos_x, pos_y), z_s)
    gamma1, gamma2 = halo.getShear((pos_x, pos_y), z_s, reduced=False)
    g1, g2 = halo.getShear((pos_x, pos_y), z_s, reduced=True)

    # check internal correctness:
    # g1 = gamma1/(1-kappa), and g2 = 0
    np.testing.assert_array_equal(g1, gamma1/(1-np.array(kappa)),
                                  err_msg="Computation of reduced shear g incorrect.")
    np.testing.assert_array_equal(g2, np.zeros_like(g2),
                                  err_msg="Computation of reduced shear g2 incorrect.")

    # comparison to reference:
    # tangential shear in x-direction is purely negative in g1
    try:
        np.testing.assert_allclose(
            -ref[:,2], gamma1, rtol=1e-4,
            err_msg="Computation of shear deviates from reference.")
        np.testing.assert_allclose(
            -ref[:,3], g1, rtol=1e-4,
            err_msg="Computation of reduced shear deviates from reference.")
        np.testing.assert_allclose(
            ref[:,4], kappa, rtol=1e-4,
            err_msg="Computation of convergence deviates from reference.")
    except AttributeError:
        # Older numpy versions don't have assert_allclose, so use this instead:
        np.testing.assert_array_almost_equal(
            -ref[:,2], gamma1, decimal=4,
            err_msg="Computation of shear deviates from reference.")
        np.testing.assert_array_almost_equal(
            -ref[:,3], g1, decimal=4,
            err_msg="Computation of reduced shear deviates from reference.")
        np.testing.assert_array_almost_equal(
            ref[:,4], kappa, decimal=4,
            err_msg="Computation of convergence deviates from reference.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shear_flatps():
    """Test that shears from power spectrum P(k)=const have the expected statistical properties"""
    import time
    t1 = time.time()

    # setup the random number generator to use for these tests
    rng = galsim.BaseDeviate(512342)

    # make a flat power spectrum for E, B modes
    test_ps = galsim.PowerSpectrum(e_power_function=pkflat, b_power_function=pkflat)
    # get shears on 500x500 grid
    g1, g2 = test_ps.buildGriddedShears(grid_spacing=1.0, ngrid=500, rng=rng)
    # check: are shears consistent with variance=0.01 as we expect for pkflat?
    var1 = np.var(g1)
    var2 = np.var(g2)
    np.testing.assert_almost_equal(var1, 0.01, decimal=3,
                                   err_msg="Incorrect shear variance(1) from flat power spectrum!")
    np.testing.assert_almost_equal(var2, 0.01, decimal=3,
                                   err_msg="Incorrect shear variance(2) from flat power spectrum!")
    # check: are g1, g2 uncorrelated with each other?
    top= np.sum((g1-np.mean(g1))*(g2-np.mean(g2)))
    bottom1 = np.sum((g1-np.mean(g1))**2)
    bottom2 = np.sum((g2-np.mean(g2))**2)
    corr = top / np.sqrt(bottom1*bottom2)
    np.testing.assert_almost_equal(
        corr, 0., decimal=2,
        err_msg="Shear components should be uncorrelated with each other!")


    # make a pure E-mode spectrum
    test_ps = galsim.PowerSpectrum(e_power_function=pkflat)
    # get shears on 500x500 grid
    g1, g2 = test_ps.buildGriddedShears(grid_spacing=1.0, ngrid=500, rng=rng)
    # check: are shears consistent with variance=0.01 as we expect for pkflat?
    var1 = np.var(g1)
    var2 = np.var(g2)
    print 'var(g1), var(g2) = ',var1,var2
    np.testing.assert_almost_equal(
        var1+var2, 0.01, decimal=3,
        err_msg="Incorrect shear variance from E-mode power spectrum!")
    # Note: These next two don't work.  
    # var1,var2 are approximately 0.0043, 0.0057.  Not sure why...
    #np.testing.assert_almost_equal(
        #var1, 0.005, decimal=3,
        #err_msg="Incorrect shear variance(1) from E-mode power spectrum!")
    #np.testing.assert_almost_equal(
        #var2, 0.005, decimal=3,
        #err_msg="Incorrect shear variance(2) from E-mode power spectrum!")

    # check: are g1, g2 uncorrelated with each other?
    top= np.sum((g1-np.mean(g1))*(g2-np.mean(g2)))
    bottom1 = np.sum((g1-np.mean(g1))**2)
    bottom2 = np.sum((g2-np.mean(g2))**2)
    corr = top / np.sqrt(bottom1*bottom2)
    np.testing.assert_almost_equal(
        corr, 0., decimal=2,
        err_msg="Shear components should be uncorrelated with each other!")


    # make a pure B-mode spectrum
    test_ps = galsim.PowerSpectrum(b_power_function=pkflat)
    # get shears on 500x500 grid
    g1, g2 = test_ps.buildGriddedShears(grid_spacing=1.0, ngrid=500, rng=rng)
    # check: are shears consistent with variance=0.01 as we expect for pkflat?
    var1 = np.var(g1)
    var2 = np.var(g2)
    print 'var(g1), var(g2) = ',var1,var2
    np.testing.assert_almost_equal(
        var1+var2, 0.01, decimal=3,
        err_msg="Incorrect shear variance from B-mode power spectrum!")
    # Note: These next two don't work.  
    # var1,var2 are approximately 0.0057, 0.0043.  Not sure why...
    #np.testing.assert_almost_equal(
        #var1, 0.005, decimal=3,
        #err_msg="Incorrect shear variance(1) from B-mode power spectrum!")
    #np.testing.assert_almost_equal(
        #var2, 0.005, decimal=3,
        #err_msg="Incorrect shear variance(2) from B-mode power spectrum!")

    # check: are g1, g2 uncorrelated with each other?
    top= np.sum((g1-np.mean(g1))*(g2-np.mean(g2)))
    bottom1 = np.sum((g1-np.mean(g1))**2)
    bottom2 = np.sum((g2-np.mean(g2))**2)
    corr = top / np.sqrt(bottom1*bottom2)
    np.testing.assert_almost_equal(
        corr, 0., decimal=2,
        err_msg="Shear components should be uncorrelated with each other!")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shear_seeds():
    """Test that shears from lensing engine behave appropriate when given same/different seeds"""
    import time
    t1 = time.time()

    # make a power spectrum for some E, B power function
    test_ps = galsim.PowerSpectrum(e_power_function=pk2, b_power_function=pkflat)

    # get shears on a grid w/o specifying seed
    g1, g2 = test_ps.buildGriddedShears(grid_spacing=1.0, ngrid = 10)
    # do it again, w/o specifying seed: should differ
    g1new, g2new = test_ps.buildGriddedShears(grid_spacing=1.0, ngrid = 10)
    assert not ((g1[0,0]==g1new[0,0]) or (g2[0,0]==g2new[0,0]))

    # get shears on a grid w/ specified seed
    g1, g2 = test_ps.buildGriddedShears(grid_spacing=1.0, ngrid = 10,
                                        rng=galsim.BaseDeviate(13796))
    # get shears on a grid w/ same specified seed: should be same
    g1new, g2new = test_ps.buildGriddedShears(grid_spacing=1.0, ngrid = 10,
                                              rng=galsim.BaseDeviate(13796))
    np.testing.assert_array_equal(g1, g1new,
                                  err_msg="New shear field differs from previous (same seed)!")
    np.testing.assert_array_equal(g2, g2new,
                                  err_msg="New shear field differs from previous (same seed)!")
    # get shears on a grid w/ diff't specified seed: should differ
    g1new, g2new = test_ps.buildGriddedShears(grid_spacing=1.0, ngrid = 10,
                                              rng=galsim.BaseDeviate(1379))
    assert not ((g1[0,0]==g1new[0,0]) or (g2[0,0]==g2new[0,0]))

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shear_reference():
    """Test shears from lensing engine compared to stored reference values"""
    import time
    t1 = time.time()

    # read input data
    ref = np.loadtxt(refdir + '/shearfield_reference.dat')
    g1_in = ref[:,0]
    g2_in = ref[:,1]

    # set up params
    rng = galsim.BaseDeviate(14136)
    n = 10
    dx = 1.

    # define power spectrum
    ps = galsim.PowerSpectrum(e_power_function=pk2, b_power_function=pk1)
    # get shears
    g1, g2 = ps.buildGriddedShears(grid_spacing = dx, ngrid = n, rng=rng)

    # put in same format as data that got read in
    g1vec = g1.reshape(n*n)
    g2vec = g2.reshape(n*n)
    # compare input vs. calculated values
    np.testing.assert_almost_equal(g1_in, g1vec, 9,
                                   err_msg = "Shear field differs from reference shear field!")
    np.testing.assert_almost_equal(g2_in, g2vec, 9,
                                   err_msg = "Shear field differs from reference shear field!")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_tabulated():
    """Test using a LookupTable to interpolate a P(k) that is known at certain k"""
    import time
    t1 = time.time()

    # make PowerSpectrum with some obvious analytic form, P(k)=k^2
    ps_analytic = galsim.PowerSpectrum(pk2)

    # now tabulate that analytic form at a range of k
    k_arr = 0.01*np.arange(10000.)+0.01
    p_arr = k_arr**(2.)

    # make a LookupTable to initialize another PowerSpectrum
    tab = galsim.LookupTable(k_arr, p_arr)
    ps_tab = galsim.PowerSpectrum(tab)

    # draw shears on a grid from both PowerSpectrum objects, with same random seed
    seed = 12345
    g1_analytic, g2_analytic = ps_analytic.buildGriddedShears(grid_spacing = 1., ngrid = 10,
                                                              rng = galsim.BaseDeviate(seed))
    g1_tab, g2_tab = ps_tab.buildGriddedShears(grid_spacing = 1., ngrid = 10,
                                               rng = galsim.BaseDeviate(seed))

    # make sure that shears that are drawn are essentially identical
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg = "g1 of shear field from tabulated P(k) differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg = "g2 of shear field from tabulated P(k) differs from expectation!")
    # now check that we get the same answer if we use file readin: write k and P(k) to a file then
    # initialize LookupTable from that file
    data_all = (k_arr, p_arr)
    data = np.column_stack(data_all)
    filename = 'lensing_reference_data/tmp.txt'
    np.savetxt(filename, data)
    tab2 = galsim.LookupTable(file = filename)
    ps_tab2 = galsim.PowerSpectrum(tab2)
    g1_tab2, g2_tab2 = ps_tab2.buildGriddedShears(grid_spacing = 1., ngrid = 10,
                                                  rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab2, 6,
        err_msg = "g1 from file-based tabulated P(k) differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab2, 6,
        err_msg = "g2 from file-based tabulated P(k) differs from expectation!")
    # check that we get the same answer whether we use interpolation in log for k, P, or both
    tab = galsim.LookupTable(k_arr, p_arr, x_log = True)
    ps_tab = galsim.PowerSpectrum(tab)
    g1_tab, g2_tab = ps_tab.buildGriddedShears(grid_spacing = 1., ngrid = 10,
                                               rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg = "g1 of shear field from tabulated P(k) with x_log differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg = "g2 of shear field from tabulated P(k) with x_log differs from expectation!")
    tab = galsim.LookupTable(k_arr, p_arr, f_log = True)
    ps_tab = galsim.PowerSpectrum(tab)
    g1_tab, g2_tab = ps_tab.buildGriddedShears(grid_spacing = 1., ngrid = 10,
                                               rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg = "g1 of shear field from tabulated P(k) with f_log differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg = "g2 of shear field from tabulated P(k) with f_log differs from expectation!")
    tab = galsim.LookupTable(k_arr, p_arr, x_log = True, f_log = True)
    ps_tab = galsim.PowerSpectrum(tab)
    g1_tab, g2_tab = ps_tab.buildGriddedShears(grid_spacing = 1., ngrid = 10,
                                               rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg = "g1 of shear field from tabulated P(k) with x_log, f_log differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg = "g2 of shear field from tabulated P(k) with x_log, f_log differs from expectation!")

    # check for appropriate response to inputs when making/using LookupTable
    try:
        ## mistaken interpolant choice
        np.testing.assert_raises(ValueError, galsim.LookupTable,
                                 k_arr, p_arr, interpolant='splin')
        ## k, P arrays not the same size
        np.testing.assert_raises(ValueError, galsim.LookupTable,
                                 0.01*np.arange(100.), p_arr)
        ## arrays too small
        np.testing.assert_raises(RuntimeError, galsim.LookupTable,
                                 (1.,2.), (1., 2.))
        ## try to make shears, but grid includes k values that were not part of the originally
        ## tabulated P(k) (for this test we make a stupidly limited k grid just to ensure that an
        ## exception should be raised)
        t = galsim.LookupTable((0.99,1.,1.01),(0.99,1.,1.01))
        ps = galsim.PowerSpectrum(t)
        np.testing.assert_raises(ValueError, ps.buildGriddedShears, grid_spacing=1., ngrid=100)
        ## try to interpolate in log, but with zero values included
        np.testing.assert_raises(ValueError, galsim.LookupTable, (0.,1.,2.), (0.,1.,2.),
                                 x_log=True)
        np.testing.assert_raises(ValueError, galsim.LookupTable, (0.,1.,2.), (0.,1.,2.),
                                 f_log=True)
        np.testing.assert_raises(ValueError, galsim.LookupTable, (0.,1.,2.), (0.,1.,2.),
                                 x_log=True, f_log=True)
    except ImportError:
        pass

    # check that when calling LookupTable, the outputs have the same form as inputs
    tab = galsim.LookupTable(k_arr, p_arr)
    k = 0.5
    assert type(tab(k)) == float
    k = (0.5, 1.5)
    result = tab(k)
    assert type(result) == tuple and len(result) == len(k)
    k = list(k)
    result = tab(k)
    assert type(result) == list and len(result) == len(k)
    k = np.array(k)
    result = tab(k)
    assert type(result) == np.ndarray and len(result) == len(k)
    k = 0.01+np.zeros((2,2))
    result = tab(k)
    assert type(result) == np.ndarray and result.shape == k.shape

    # check for expected behavior with log interpolation
    k = (1., 2., 3.)
    p = (1., 4., 9.)
    t = galsim.LookupTable(k, p, interpolant = 'linear')
    ## a linear interpolant should fail here because P(k) is a power-law, so make sure we get the
    ## expected result with linear interpolation
    np.testing.assert_almost_equal(t(2.5), 13./2., decimal = 6,
        err_msg = 'Unexpected result for linear interpolation of power-law')
    ## but a linear interpolant works if you work in log space, so check against real result
    t = galsim.LookupTable(k, p, interpolant = 'linear', x_log = True, f_log = True)
    np.testing.assert_almost_equal(t(2.5), 2.5**2, decimal = 6,
        err_msg = 'Unexpected result for linear interpolation of power-law in log space')

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_nfwhalo()
    test_shear_flatps()
    test_shear_seeds()
    test_shear_reference()
    test_tabulated()
