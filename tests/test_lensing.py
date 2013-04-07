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
import sys

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

refdir = os.path.join(".", "lensing_reference_data") # Directory containing the reference

klim_test = 0.00175 # Value of klim for flat (up to klim, then zero beyond) power spectrum test


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
    parr[k<=klim_test] = 1.
    return parr

def kappa_gaussian(theta1_array, theta2_array, sigma, pos, amp=1.):
    """Return an array of kappa values given input arrays of theta1, theta2, float for sigma in the
    units of theta and a galsim.PositionD for pos.

    Scaled to be amp at origin.
    """
    sigma2 = sigma * sigma
    theta2 = (theta1_array - pos.x)**2 + (theta2_array - pos.y)**2
    return amp * np.exp(-.5 * theta2 / sigma2)

def shear_gaussian(theta1_array, theta2_array, sigma, pos, amp=1., rotate45=False):
    """Return arrays of shear values given input arrays of theta1, theta2, float for sigma in the
    units of theta and a galsim.PositionD for pos.

    The surface density kappa is scaled to be amp at origin.  Tangential shear around the origin
    relates to kappa_gaussian function by the usual relation,
    gamma_t(theta) = <kappa(<theta)> - kappa(theta)
    The rotate45 keyword can be used to specify rotation of the shears at each point by 45 degrees.
    """
    sigma2 = sigma * sigma
    t1 = theta1_array - pos.x
    t2 = theta2_array - pos.y
    theta2 = t1 * t1 + t2 * t2
    gammat = 2. * amp * sigma2 * (
        1. - (1. + .5 * theta2 / sigma2) * np.exp(-.5 * theta2 / sigma2)) / theta2 
    if rotate45:
        g2 = -gammat * (t1**2 - t2**2) / theta2
        g1 =  gammat * 2. * t1 * t2 / theta2
    else:
        g1 = -gammat * (t1**2 - t2**2) / theta2
        g2 = -gammat * 2. * t1 * t2 / theta2
    return g1, g2

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

def test_shear_variance():
    """Test that shears from several toy power spectra have the expected variances."""
    import time
    t1 = time.time()

    # setup the random number generator to use for these tests
    rng = galsim.BaseDeviate(512342)

    # set up grid parameters
    grid_size = 50. # degrees
    ngrid = 500 # grid points
    klim = klim_test
    # now get derived grid parameters
    kmin = 2.*np.pi/grid_size/3600. # arcsec^-1

    # Make a flat power spectrum for E, B modes with P=1 arcsec^2, truncated above some limiting
    # value of k.
    # Given our grid size of 50 degrees [which is silly to do for a flat-sky approximation, but
    # we're just doing it anyway to beat down the noise], the minimum k we can probe is 2pi/50
    # deg^{-1} = 3.49e-5 arcsec^-1.  With 500 grid points, the maximum k in one dimension is 250
    # times as large, 0.00873 arcsec^-1.  The function pk_flat_lim is 0 for k>klim_test=0.00175 which
    # is a factor of 5 below our maximum k, a factor of ~50 above our minimum k.  For k<=0.00175,
    # pk_flat_lim returns 1.
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim, b_power_function=pk_flat_lim)
    # get shears on 500x500 grid with spacing 0.1 degree
    g1, g2 = test_ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
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
    np.testing.assert_almost_equal(comparison_val/2., 0., decimal=2,
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
    g1, g2 = test_ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                               units=galsim.degrees)
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    var1 = np.var(g1)
    var2 = np.var(g2)
    comparison_val = (var1+var2)/(0.985*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/2., 0., decimal=2,
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
    klim = klim_test
    kmin = 2.*np.pi/grid_size/3600. # arcsec^-1
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim, b_power_function=pk_flat_lim)
    g1, g2 = test_ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                               units=galsim.degrees)
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    var1 = np.var(g1)
    var2 = np.var(g2)
    comparison_val = (var1+var2)/(0.985*2.*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/2., 0., decimal=2,
                                   err_msg="Incorrect shear variance from flat power spectrum!")

    # check for proper scaling with number of grid points, for fixed grid spacing
    grid_size = 25. # degrees
    ngrid = 250 # grid points
    klim = klim_test 
    kmin = 2.*np.pi/grid_size/3600. # arcsec^-1
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim, b_power_function=pk_flat_lim)
    g1, g2 = test_ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
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
    kmax = np.pi/(grid_size/ngrid)/3600.
    # For explanation of these two variables, see below, the comment starting "Note: the next..."
    # These numbers are, however, hard-coded up here with the grid parameters because if the grid is
    # changed, the erfmax and erfmin must change.
    erfmax = 0.9875806693484477
    erfmin = 0.007978712629263206
    # Now choose s such that s*kmax=2.5, i.e., very little power at kmax.
    s = 2.5/kmax
    test_ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    g1, g2 = test_ps.buildGrid(grid_spacing = grid_size/ngrid, ngrid=ngrid, rng=rng,
                               units=galsim.degrees)
    # For this case, the prediction for the variance is:
    # Var(g1) + Var(g2) = [1/(2 pi s^2)] * ( (Erf(s*kmax/sqrt(2)))^2 - (Erf(s*kmin/sqrt(2)))^2 )
    # Note: the next two lines of code are commented out because math.erf is not available in python
    # v2.6, with which we must be compatible.  So instead the values of erf are hard-coded (above)
    # based on the calculations from a machine that has python v2.7.  The implications here are that
    # if one changes the grid parameters for this test in a way that also changes the values of
    # these Erf[...] calculations, then the hard-coded erfmax and erfmin must be changed.
    # erfmax = math.erf(s*kmax/math.sqrt(2.))
    # erfmin = math.erf(s*kmin/math.sqrt(2.))
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    # here we know that the results are typically 2.5% too low, and we again allow wiggle room of 3.5%
    # due to noise.
    comparison_val = (np.var(g1)+np.var(g2))/(0.975*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/4.5, 0., decimal=2,
                                   err_msg="Incorrect variance from Gaussian PS")

    # check for proper scaling with grid spacing, for fixed number of grid points
    grid_size = 25. # degrees
    ngrid = 500 # grid points
    kmin = 2.*np.pi/grid_size/3600.
    kmax = np.pi/(grid_size/ngrid)/3600.
    s = 2.5/kmax
    # Note that because of how s, kmin, and kmax change, the Erf[...] quantities do not change.  So
    # we don't have to reset the values here.
    test_ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    g1, g2 = test_ps.buildGrid(grid_spacing = grid_size/ngrid, ngrid=ngrid,
                               rng=rng, units=galsim.degrees)
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    comparison_val = (np.var(g1)+np.var(g2))/(0.975*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/5., 0., decimal=2,
                                   err_msg="Incorrect variance from Gaussian PS")

    # check for proper scaling with number of grid points, for fixed grid spacing
    grid_size = 25. # degrees
    ngrid = 250 # grid points
    kmin = 2.*np.pi/grid_size/3600.
    kmax = np.pi/(grid_size/ngrid)/3600.
    # Here one of the Erf[...] values does change.
    erfmin = 0.01595662743380396
    s = 2.5/kmax
    test_ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    g1, g2 = test_ps.buildGrid(grid_spacing = grid_size/ngrid, ngrid=ngrid,
                               rng=rng, units=galsim.degrees)
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    comparison_val = (np.var(g1)+np.var(g2))/(0.975*predicted_variance)-1.0
    np.testing.assert_almost_equal(comparison_val/4.5, 0., decimal=2,
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
    g1, g2 = test_ps.buildGrid(grid_spacing=1.0, ngrid = 10)
    # do it again, w/o specifying seed: should differ
    g1new, g2new = test_ps.buildGrid(grid_spacing=1.0, ngrid = 10)
    assert not ((g1[0,0]==g1new[0,0]) or (g2[0,0]==g2new[0,0]))

    # get shears on a grid w/ specified seed
    g1, g2 = test_ps.buildGrid(grid_spacing=1.0, ngrid = 10, rng=galsim.BaseDeviate(13796))
    # get shears on a grid w/ same specified seed: should be same
    g1new, g2new = test_ps.buildGrid(grid_spacing=1.0, ngrid = 10, rng=galsim.BaseDeviate(13796))
    np.testing.assert_array_equal(g1, g1new,
                                  err_msg="New shear field differs from previous (same seed)!")
    np.testing.assert_array_equal(g2, g2new,
                                  err_msg="New shear field differs from previous (same seed)!")
    # get shears on a grid w/ diff't specified seed: should differ
    g1new, g2new = test_ps.buildGrid(grid_spacing=1.0, ngrid = 10, rng=galsim.BaseDeviate(1379))
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
    kappa_in = ref[:,2]

    # set up params
    rng = galsim.BaseDeviate(14136)
    n = 10
    dx = 1.

    # define power spectrum
    ps = galsim.PowerSpectrum(e_power_function=lambda k : k**0.5, b_power_function=lambda k : k)
    # get shears
    g1, g2, kappa = ps.buildGrid(grid_spacing = dx, ngrid = n, rng=rng, get_convergence=True)

    # put in same format as data that got read in
    g1vec = g1.reshape(n*n)
    g2vec = g2.reshape(n*n)
    kappavec = kappa.reshape(n*n)
    # compare input vs. calculated values
    np.testing.assert_almost_equal(g1_in, g1vec, 9,
                                   err_msg = "Shear field differs from reference shear field!")
    np.testing.assert_almost_equal(g2_in, g2vec, 9,
                                   err_msg = "Shear field differs from reference shear field!")
    np.testing.assert_almost_equal(kappa_in, kappavec, 9,
                                   err_msg = "Convergence differences from references!")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_shear_get():
    """Check that using gridded outputs and the various getFoo methods gives consistent results"""
    import time
    t1 = time.time()

    # choose a power spectrum and grid setup
    my_ps = galsim.PowerSpectrum(lambda k : k**0.5)
    # build the grid
    grid_spacing = 1.
    ngrid = 100
    g1, g2, kappa = my_ps.buildGrid(grid_spacing = grid_spacing, ngrid = ngrid,
                                    get_convergence = True)
    min = (-ngrid/2 + 0.5) * grid_spacing
    max = (ngrid/2 - 0.5) * grid_spacing
    x, y = np.meshgrid(np.arange(min,max+grid_spacing,grid_spacing),
                       np.arange(min,max+grid_spacing,grid_spacing))

    # convert theoretical to observed quantities for grid
    g1_r, g2_r, mu = galsim.lensing.theoryToObserved(g1, g2, kappa)

    # use getShear, getConvergence, getMagnification, getLensing do appropriate consistency checks
    test_g1_r, test_g2_r = my_ps.getShear((x.flatten(), y.flatten()))
    test_g1, test_g2 = my_ps.getShear((x.flatten(), y.flatten()), reduced = False)
    test_kappa = my_ps.getConvergence((x.flatten(), y.flatten()))
    test_mu = my_ps.getMagnification((x.flatten(), y.flatten()))
    test_g1_r_2, test_g2_r_2, test_mu_2 = my_ps.getLensing((x.flatten(), y.flatten()))
    np.testing.assert_almost_equal(g1.flatten(), test_g1, 9,
                                   err_msg="Shears from grid and getShear disagree!")
    np.testing.assert_almost_equal(g2.flatten(), test_g2, 9,
                                   err_msg="Shears from grid and getShear disagree!")
    np.testing.assert_almost_equal(g1_r.flatten(), test_g1_r, 9,
                                   err_msg="Reduced shears from grid and getShear disagree!")
    np.testing.assert_almost_equal(g2_r.flatten(), test_g2_r, 9,
                                   err_msg="Reduced shears from grid and getShear disagree!")
    np.testing.assert_almost_equal(g1_r.flatten(), test_g1_r_2, 9,
                                   err_msg="Reduced shears from grid and getLensing disagree!")
    np.testing.assert_almost_equal(g2_r.flatten(), test_g2_r_2, 9,
                                   err_msg="Reduced shears from grid and getLensing disagree!")
    np.testing.assert_almost_equal(kappa.flatten(), test_kappa, 9,
                                   err_msg="Convergences from grid and getConvergence disagree!")
    np.testing.assert_almost_equal(mu.flatten(), test_mu, 9,
                                   err_msg="Magnifications from grid and getMagnification disagree!")
    np.testing.assert_almost_equal(mu.flatten(), test_mu_2, 9,
                                   err_msg="Magnifications from grid and getLensing disagree!")

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
    g1, g2 = ps.buildGrid(grid_spacing = 3600.*grid_size/ngrid, ngrid=ngrid,
                          rng = galsim.BaseDeviate(rand_seed))
    # The above was done with all inputs given in arcsec.  Now, redo it, inputting the PS
    # information in degrees and the grid info in arcsec.
    # We know that if k=1/arcsec, then when expressed as 1/degrees, it is
    # k=3600/degree.  So define the PS as P(k=3600/degree)=(1/3600.)^2 degree^2.
    ps = galsim.PowerSpectrum(lambda k : (1./3600.**2)*(k/3600.), units=galsim.degrees)
    g1_2, g2_2 = ps.buildGrid(grid_spacing = 3600.*grid_size/ngrid, ngrid=ngrid,
                              rng=galsim.BaseDeviate(rand_seed))
    # Finally redo it, inputting the PS and grid info in degrees.
    ps = galsim.PowerSpectrum(lambda k : (1./3600.**2)*(k/3600.), units=galsim.degrees)
    g1_3, g2_3 = ps.buildGrid(grid_spacing = grid_size/ngrid, ngrid=ngrid,
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
    g1_analytic, g2_analytic = ps_analytic.buildGrid(grid_spacing = 1., ngrid = 10,
                                                     rng = galsim.BaseDeviate(seed))
    g1_tab, g2_tab = ps_tab.buildGrid(grid_spacing = 1., ngrid = 10, rng = galsim.BaseDeviate(seed))

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
    g1_tab2, g2_tab2 = ps_tab2.buildGrid(grid_spacing = 1., ngrid = 10,
                                         rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab2, 6,
        err_msg = "g1 from file-based tabulated P(k) differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab2, 6,
        err_msg = "g2 from file-based tabulated P(k) differs from expectation!")
    # check that we get the same answer whether we use interpolation in log for k, P, or both
    tab = galsim.LookupTable(k_arr, p_arr, x_log = True)
    ps_tab = galsim.PowerSpectrum(tab)
    g1_tab, g2_tab = ps_tab.buildGrid(grid_spacing = 1., ngrid = 10, rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg = "g1 of shear field from tabulated P(k) with x_log differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg = "g2 of shear field from tabulated P(k) with x_log differs from expectation!")
    tab = galsim.LookupTable(k_arr, p_arr, f_log = True)
    ps_tab = galsim.PowerSpectrum(tab)
    g1_tab, g2_tab = ps_tab.buildGrid(grid_spacing = 1., ngrid = 10, rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg = "g1 of shear field from tabulated P(k) with f_log differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg = "g2 of shear field from tabulated P(k) with f_log differs from expectation!")
    tab = galsim.LookupTable(k_arr, p_arr, x_log = True, f_log = True)
    ps_tab = galsim.PowerSpectrum(tab)
    g1_tab, g2_tab = ps_tab.buildGrid(grid_spacing = 1., ngrid = 10, rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg="g1 of shear field from tabulated P(k) with x_log, f_log differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg="g2 of shear field from tabulated P(k) with x_log, f_log differs from expectation!")

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
        np.testing.assert_raises(ValueError, ps.buildGrid, grid_spacing=1., ngrid=100)
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

def test_kappa_gauss():
    """Test that our Kaiser-Squires inversion routine correctly recovers the convergence map
    for a field containing known Gaussian-profile halos.
    """
    import time
    t1 = time.time()
    # Setup coordinates for gridded kappa/gamma
    # Note: if the grid extent is made larger (i.e. the image is zero padded) then accuracy on
    # the output kappa map is increased, and decimal can be increased too.  We confirmed this
    # behavior, but do not use the higher resolution for unit tests due to the inefficiency of
    # calculations on larger grids.
    grid_spacing_arcsec = 1
    grid_extent_arcsec = 100.
    ngrid = int(grid_extent_arcsec / grid_spacing_arcsec)
    grid_side = (np.arange(ngrid, dtype=float) + .5) * grid_spacing_arcsec - .5 * grid_extent_arcsec
    x, y = np.meshgrid(grid_side, grid_side)
    # Get the kappas by putting two Gaussian halos in the field
    k_big = kappa_gaussian(x, y, sigma=5., pos=galsim.PositionD(-6., -6.), amp=.5)
    k_sml = kappa_gaussian(x, y, sigma=4., pos=galsim.PositionD(6., 6.), amp=.2)
    # Get the shears for the same halos
    g1_big, g2_big = shear_gaussian(x, y, sigma=5., pos=galsim.PositionD(-6., -6.), amp=.5)
    g1_sml, g2_sml = shear_gaussian(x, y, sigma=4., pos=galsim.PositionD(6., 6.), amp=.2)
    # Combine the big and small halos into the field
    g1 = g1_big + g1_sml
    g2 = g2_big + g2_sml
    # Get the reference kappa
    k_ref = k_big + k_sml
    # Invert to get the test kappa
    k_testE, k_testB = galsim.lensing.kappaKaiserSquires(g1, g2)
    # Then run tests based on the central region to avoid edge effects (known issue with KS
    # inversion)
    icent = np.arange(ngrid / 2) + ngrid / 4
    # Test that E-mode kappa matches
    np.testing.assert_array_almost_equal(
        k_testE[np.ix_(icent, icent)], k_ref[np.ix_(icent, icent)], decimal=2,
        err_msg="Reconstructed kappa does not match input to 2 decimal places.")
    # Test B-mode kappa is consistent with zero
    np.testing.assert_array_almost_equal(
        k_testB[np.ix_(icent, icent)], np.zeros((ngrid / 2, ngrid / 2)), decimal=3,
        err_msg="Reconstructed B-mode kappa is non-zero at greater than 3 decimal places.")
    # Generate shears using the 45 degree rotation option
    g1r_big, g2r_big = shear_gaussian(
        x, y, sigma=5., pos=galsim.PositionD(-6., -6.), amp=.5, rotate45=True)
    g1r_sml, g2r_sml = shear_gaussian(
        x, y, sigma=4., pos=galsim.PositionD(6., 6.), amp=.2, rotate45=True)
    g1r = g1r_big + g1r_sml
    g2r = g2r_big + g2r_sml
    kr_testE, kr_testB = galsim.lensing.kappaKaiserSquires(g1r, g2r)
    # Test that B-mode kappa for rotated shear field matches E mode
    np.testing.assert_array_almost_equal(
        kr_testB[np.ix_(icent, icent)], k_ref[np.ix_(icent, icent)], decimal=2,
        err_msg="Reconstructed kappaB in rotated shear field does not match original kappaE to 2 "+
        "decimal places.")
    # Test E-mode kappa is consistent with zero for rotated shear field
    np.testing.assert_array_almost_equal(
        kr_testE[np.ix_(icent, icent)], np.zeros((ngrid / 2, ngrid / 2)), decimal=3,
        err_msg="Reconstructed kappaE is non-zero at greater than 3 decimal places for rotated "+
        "shear field.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

def test_power_spectrum_with_kappa():
    """Test that the convergence map generated by the PowerSpectrum class is consistent with the
    Kaiser Squires inversion of the corresponding shear field.
    """
    import time
    t1 = time.time()
    # Note that in order for this test to pass, we have to control aliasing by smoothing the power
    # spectrum to go to zero above some maximum k.  This is the only way to get agreement at high
    # precision between the gamma's and kappa's from the lensing engine vs. that from a Kaiser and
    # Squires inversion.
    rseed=177774
    ngrid=100
    dx_grid_arcmin = 6
    # First lookup a cosmologically relevant power spectrum (bandlimited version to remove aliasing
    # and allow high-precision comparison).
    tab_ps = galsim.LookupTable(
        file='../examples/data/cosmo-fid.zmed1.00_smoothed.out', interpolant='linear')

    # Begin with E-mode input power
    psE = galsim.PowerSpectrum(tab_ps, None, units=galsim.radians)
    g1E, g2E, k_test = psE.buildGrid(
        grid_spacing=dx_grid_arcmin, ngrid=ngrid, units=galsim.arcmin,
        rng=galsim.BaseDeviate(rseed), get_convergence=True)
    kE_ks, kB_ks = galsim.lensing.kappaKaiserSquires(g1E, g2E)
    # Test that E-mode kappa matches to some sensible accuracy
    exact_dp = 15
    np.testing.assert_array_almost_equal(
        k_test, kE_ks, decimal=exact_dp,
        err_msg="E-mode only PowerSpectrum output kappaE does not match KS inversion to 16 d.p.")
    # Test that B-mode kappa matches zero to some sensible accuracy
    np.testing.assert_array_almost_equal(
        kB_ks, np.zeros_like(kE_ks), decimal=exact_dp,
        err_msg="E-mode only PowerSpectrum output kappaB from KS does not match zero to 16 d.p.")

    # Then do B-mode only input power
    psB = galsim.PowerSpectrum(None, tab_ps, units=galsim.radians)
    g1B, g2B, k_test = psB.buildGrid(
        grid_spacing=dx_grid_arcmin, ngrid=ngrid, units=galsim.arcmin,
        rng=galsim.BaseDeviate(rseed), get_convergence=True)
    kE_ks, kB_ks = galsim.lensing.kappaKaiserSquires(g1B, g2B)
    # Test that kappa output by PS code matches zero to some sensible accuracy
    np.testing.assert_array_almost_equal(
        k_test, np.zeros_like(k_test), decimal=exact_dp,
        err_msg="B-mode only PowerSpectrum output kappa does not match zero to 16 d.p.")
    # Test that E-mode kappa inferred via KS also matches zero to some sensible accuracy
    np.testing.assert_array_almost_equal(
        kE_ks, np.zeros_like(kB_ks), decimal=exact_dp,
        err_msg="B-mode only PowerSpectrum output kappaE from KS does not match zero to 16 d.p.")

    # Then for luck take B-mode only shears but rotate by 45 degrees before KS, and check
    # consistency 
    kE_ks_rotated, kB_ks_rotated = galsim.lensing.kappaKaiserSquires(g2B, -g1B)
    np.testing.assert_array_almost_equal(
        kE_ks_rotated, kB_ks, decimal=exact_dp,
        err_msg="KS inverted kappaE from B-mode only PowerSpectrum fails rotation test.")
    np.testing.assert_array_almost_equal(
        kB_ks_rotated, np.zeros_like(kB_ks), decimal=exact_dp,
        err_msg="KS inverted kappaB from B-mode only PowerSpectrum fails rotation test.")

    # Finally, do E- and B-mode power
    psB = galsim.PowerSpectrum(tab_ps, tab_ps, units=galsim.radians)
    g1EB, g2EB, k_test = psB.buildGrid(
        grid_spacing=dx_grid_arcmin, ngrid=ngrid, units=galsim.arcmin,
        rng=galsim.BaseDeviate(rseed), get_convergence=True)
    kE_ks, kB_ks = galsim.lensing.kappaKaiserSquires(g1EB, g2EB)
    # Test that E-mode kappa matches to some sensible accuracy
    np.testing.assert_array_almost_equal(
        k_test, kE_ks, decimal=exact_dp,
        err_msg="E/B PowerSpectrum output kappa does not match KS inversion to 16 d.p.")
    # Test rotating the shears by 45 degrees
    kE_ks_rotated, kB_ks_rotated = galsim.lensing.kappaKaiserSquires(g2EB, -g1EB)
    np.testing.assert_array_almost_equal(
        kE_ks_rotated, kB_ks, decimal=exact_dp,
        err_msg="KS inverted kappaE from E/B PowerSpectrum fails rotation test.")
    np.testing.assert_array_almost_equal(
        kB_ks_rotated, -kE_ks, decimal=exact_dp,
        err_msg="KS inverted kappaB from E/B PowerSpectrum fails rotation test.")

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)

if __name__ == "__main__":
    test_nfwhalo()
    test_shear_variance()
    test_shear_seeds()
    test_shear_reference()
    test_shear_units()
    test_shear_get()
    test_tabulated()
    test_kappa_gauss()
    test_power_spectrum_with_kappa()
