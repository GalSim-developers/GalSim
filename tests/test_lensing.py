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
import math
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
def pk1(k):
    return k

def pk2(k):
    return k**(2.0)

def pk_flat_lim(k):
    parr = np.zeros_like(k)
    parr[k<=0.00175] = 1.
    return parr

def test_nfwhalo():
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

def test_shear_variance():
    """Test that shears from several toy power spectra have the expected variances."""
    import time
    t1 = time.time()

    # setup the random number generator to use for these tests
    rng = galsim.BaseDeviate(512342)

    # set up grid parameters
    grid_size = 50. # degrees
    ngrid = 500 # grid points
    klim = 0.00175 # this was hard-coded in pk_flat_lim, don't change without changing it there too
    # now get derived grid parameters
    kmin = 2.*np.pi/grid_size/3600. # arcsec^-1

    # Make a flat power spectrum for E, B modes with P=1 arcsec^2, truncated above some limiting
    # value of k.
    # Given our grid size of 50 degrees [which is silly to do for a flat-sky approximation, but
    # we're just doing it anyway to beat down the noise], the minimum k we can probe is 2pi/50
    # deg^{-1} = 3.49e-5 arcsec^-1.  With 500 grid points, the maximum k in one dimension is 500
    # times as large, 0.0175 arcsec^-1.  The function pk_flat_lim is 0 for k>0.00175 which is a
    # factor of 10 below our maximum k, a factor of ~50 above our minimum k.  For k<=0.00175,
    # pk_flat_lim returns 1.
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim, b_power_function=pk_flat_lim)
    # get shears on 500x500 grid with spacing 0.1 degree
    g1, g2 = test_ps.buildGriddedShears(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                                        units=galsim.degrees)
    # Now we should compare the variance with the predictions.  We use
    # ../devel/modules/lensing_engine.pdf section 5.3 to get
    # Var(g1) + Var(g2) = (1/pi^2) [(pi klim^2 / 4) - kmin^2]
    # Here the 1 on top of the pi^2 is actually P0 which has units of arcsec^2.
    # A final point for this test is that the result should be twice as large than that prediction
    # since we have both E and B power.  And we know from before that due to various effects, the
    # results should actually be ~1.5% too low.  So take the sum of the variances per component,
    # compare with predictions*0.985, subtract 1.  The result should be 0, but we allow a tolerance
    # of +/-2% due to noise, dividing it by 3 and requiring consistency with 0 at 2 decimal places.
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    var1 = np.var(g1)
    var2 = np.var(g2)
    comparison_val = (var1+var2)/(0.985*2.*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/3., 0., decimal=2,
                                   err_msg="Incorrect shear variance from flat power spectrum!")
    # check: are g1, g2 uncorrelated with each other?
    top= np.sum((g1-np.mean(g1))*(g2-np.mean(g2)))
    bottom1 = np.sum((g1-np.mean(g1))**2)
    bottom2 = np.sum((g2-np.mean(g2))**2)
    corr = top / np.sqrt(bottom1*bottom2)
    np.testing.assert_almost_equal(
        corr, 0., decimal=1,
        err_msg="Shear components should be uncorrelated with each other!")

    # Now do the same test as previously, but with E-mode power only.
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim)
    g1, g2 = test_ps.buildGriddedShears(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                                        units=galsim.degrees)
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    var1 = np.var(g1)
    var2 = np.var(g2)
    comparison_val = (var1+var2)/(0.985*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/3., 0., decimal=2,
                                   err_msg="Incorrect shear variance from flat power spectrum!")
    # check: are g1, g2 uncorrelated with each other?
    top= np.sum((g1-np.mean(g1))*(g2-np.mean(g2)))
    bottom1 = np.sum((g1-np.mean(g1))**2)
    bottom2 = np.sum((g2-np.mean(g2))**2)
    corr = top / np.sqrt(bottom1*bottom2)
    np.testing.assert_almost_equal(
        corr, 0., decimal=1,
        err_msg="Shear components should be uncorrelated with each other!")

    # check for proper scaling with grid spacing, for fixed number of grid points
    grid_size = 25. # degrees
    ngrid = 500 # grid points
    klim = 0.00175 # this was hard-coded in pk_flat_lim, don't change without changing it there too
    kmin = 2.*np.pi/grid_size/3600. # arcsec^-1
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim, b_power_function=pk_flat_lim)
    g1, g2 = test_ps.buildGriddedShears(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                                        units=galsim.degrees)
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    var1 = np.var(g1)
    var2 = np.var(g2)
    comparison_val = (var1+var2)/(0.985*2.*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/3., 0., decimal=2,
                                   err_msg="Incorrect shear variance from flat power spectrum!")

    # check for proper scaling with number of grid points, for fixed grid spacing
    grid_size = 25. # degrees
    ngrid = 250 # grid points
    klim = 0.00175 # this was hard-coded in pk_flat_lim, don't change without changing it there too
    kmin = 2.*np.pi/grid_size/3600. # arcsec^-1
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim, b_power_function=pk_flat_lim)
    g1, g2 = test_ps.buildGriddedShears(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                                        units=galsim.degrees)
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    var1 = np.var(g1)
    var2 = np.var(g2)
    comparison_val = (var1+var2)/(0.985*2.*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/3., 0., decimal=2,
                                   err_msg="Incorrect shear variance from flat power spectrum!")

    # Test one other theoretical PS: the Gaussian P(k).
    # We define it as P(k) = exp(-s^2 k^2 / 2).
    # First set up the grid.
    grid_size = 50. # degrees
    ngrid = 500 # grid points
    kmin = 2.*np.pi/grid_size/3600.
    kmax = 2.*np.pi/(grid_size/ngrid)/3600.
    # Now choose s such that s*kmax=5, i.e., almost no power at kmax.
    s = 5./kmax
    test_ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    g1, g2 = test_ps.buildGriddedShears(grid_spacing = grid_size/ngrid, ngrid=ngrid, rng=rng,
                                        units=galsim.degrees)
    # For this case, the prediction for the variance is:
    # Var(g1) + Var(g2) = [1/(2 pi s^2)] * ( (Erf(s*kmax/sqrt(2)))^2 - (Erf(s*kmin/sqrt(2)))^2 )
    erfmax = math.erf(s*kmax/math.sqrt(2.))
    erfmin = math.erf(s*kmin/math.sqrt(2.))
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    # here we know that the results are typically 2.5% too low, and we again allow wiggle room of 3%
    # due to noise.
    comparison_val = (np.var(g1)+np.var(g2))/(0.975*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/3., 0., decimal=2,
                                   err_msg="Incorrect variance from Gaussian PS")
    # check for proper scaling with grid spacing, for fixed number of grid points
    grid_size = 25. # degrees
    ngrid = 500 # grid points
    kmin = 2.*np.pi/grid_size/3600.
    kmax = 2.*np.pi/(grid_size/ngrid)/3600.
    s = 5./kmax
    test_ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    g1, g2 = test_ps.buildGriddedShears(grid_spacing = grid_size/ngrid, ngrid=ngrid,
                                        rng=rng, units=galsim.degrees)
    erfmax = math.erf(s*kmax/math.sqrt(2.))
    erfmin = math.erf(s*kmin/math.sqrt(2.))
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    comparison_val = (np.var(g1)+np.var(g2))/(0.975*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/3., 0., decimal=2,
                                   err_msg="Incorrect variance from Gaussian PS")
    # check for proper scaling with number of grid points, for fixed grid spacing
    grid_size = 25. # degrees
    ngrid = 250 # grid points
    kmin = 2.*np.pi/grid_size/3600.
    kmax = 2.*np.pi/(grid_size/ngrid)/3600.
    s = 5./kmax
    test_ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    g1, g2 = test_ps.buildGriddedShears(grid_spacing = grid_size/ngrid, ngrid=ngrid,
                                        rng=rng, units=galsim.degrees)
    erfmax = math.erf(s*kmax/math.sqrt(2.))
    erfmin = math.erf(s*kmin/math.sqrt(2.))
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    comparison_val = (np.var(g1)+np.var(g2))/(0.975*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/3., 0., decimal=2,
                                   err_msg="Incorrect variance from Gaussian PS")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shear_seeds():
    """Test that shears from lensing engine behave appropriate when given same/different seeds"""
    import time
    t1 = time.time()

    # make a power spectrum for some E, B power function
    test_ps = galsim.PowerSpectrum(e_power_function=pk2, b_power_function=pk2)

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

def test_shear_units():
    """Test that the shears we get out do not depend on the input PS and grid units."""
    import time
    t1 = time.time()

    rand_seed = 123456

    grid_size = 10. # degrees
    ngrid = 100

    # Define a PS with some normalization value P(k=1/arcsec)=1 arcsec^2.
    # For this case we are getting the shears using units of arcsec for everything.
    ps = galsim.PowerSpectrum(lambda k : k)
    g1, g2 = ps.buildGriddedShears(grid_spacing = 3600.*grid_size/ngrid, ngrid=ngrid,
                                   rng = galsim.BaseDeviate(rand_seed))
    # The above was done with all inputs given in arcsec.  Now, redo it, inputting the PS
    # information in degrees and the grid info in arcsec.
    # We know that if k=1/arcsec, then when expressed as 1/degrees, it is
    # k=3600/degree.  So define the PS as P(k=3600/degree)=(1/3600.)^2 degree^2.
    ps = galsim.PowerSpectrum(lambda k : (1./3600.**2)*(k/3600.), units=galsim.degrees)
    g1_2, g2_2 = ps.buildGriddedShears(grid_spacing = 3600.*grid_size/ngrid, ngrid=ngrid,
                                           rng=galsim.BaseDeviate(rand_seed))
    # Finally redo it, inputting the PS and grid info in degrees.
    ps = galsim.PowerSpectrum(lambda k : (1./3600.**2)*(k/3600.), units=galsim.degrees)
    g1_3, g2_3 = ps.buildGriddedShears(grid_spacing = grid_size/ngrid, ngrid=ngrid,
                                       units = galsim.degrees, rng=galsim.BaseDeviate(rand_seed))

    # Since same random seed was used, require complete equality of shears, which would show that
    # all unit conversions were properly handled.
    np.testing.assert_array_almost_equal(g1, g1_2, decimal=9,
                                         err_msg='Incorrect unit handling in lensing engine')
    np.testing.assert_array_almost_equal(g1, g1_3, decimal=9,
                                         err_msg='Incorrect unit handling in lensing engine')
    np.testing.assert_array_almost_equal(g2, g2_2, decimal=9,
                                         err_msg='Incorrect unit handling in lensing engine')
    np.testing.assert_array_almost_equal(g2, g2_3, decimal=9,
                                         err_msg='Incorrect unit handling in lensing engine')
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
    test_shear_variance()
    test_shear_seeds()
    test_shear_reference()
    test_shear_units()
    test_tabulated()
