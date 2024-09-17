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

import numpy as np
import math
import os

import galsim
from galsim_test_helpers import *


refdir = os.path.join(".", "lensing_reference_data") # Directory containing the reference

klim_test = 0.00175 # Value of klim for flat (up to klim, then zero beyond) power spectrum test
tolerance_var = 0.03 # fractional error allowed in the variance of shear - calculation is not exact
                     # so do not be too stringent

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


@timer
def test_nfwhalo():
    """Various tests of the NFWHalo class (against reference data, and basic sanity tests)"""
    # reference data comes from Matthias Bartelmann's libastro code
    # cluster properties: M=1e15, conc=4, redshift=1
    # sources at redshift=2
    # columns:
    # distance [arcsec], deflection [arcsec], shear, reduced shear, convergence
    # distance go from 1 .. 599 arcsec
    ref = np.loadtxt(refdir + '/nfw_lens.dat')

    # set up the same halo
    halo = galsim.NFWHalo(mass=1e15, conc=4, redshift=1)
    pos_x = np.arange(1.,600)
    pos_y = np.zeros_like(pos_x)
    z_s = 2
    # Along the way test different allowed ways to pass in position information.
    kappa = halo.getConvergence((pos_x, pos_y), z_s)
    gamma1, gamma2 = halo.getShear([pos_x, pos_y], z_s, reduced=False)
    g1, g2 = halo.getShear([galsim.PositionD(x,y) for x,y in zip(pos_x,pos_y)], z_s, reduced=True)
    mu = halo.getMagnification((pos_x, pos_y), z_s)
    alt_g1, alt_g2, alt_mu = halo.getLensing((pos_x, pos_y), z_s)

    # check internal correctness:
    # g1 = gamma1/(1-kappa), and g2 = 0
    np.testing.assert_array_equal(g1, gamma1/(1-kappa),
                                  err_msg="Computation of reduced shear g incorrect.")
    np.testing.assert_array_equal(g2, np.zeros_like(g2),
                                  err_msg="Computation of reduced shear g2 incorrect.")
    np.testing.assert_array_equal(mu, 1./( (1-kappa)**2 - gamma1**2 - gamma2**2 ),
                                  err_msg="Computation of magnification mu incorrect.")
    np.testing.assert_array_equal(alt_g1, g1,
                                  err_msg="getLensing returned wrong g1")
    np.testing.assert_array_equal(alt_g2, g2,
                                  err_msg="getLensing returned wrong g2")
    np.testing.assert_array_equal(alt_mu, mu,
                                  err_msg="getLensing returned wrong mu")

    check_pickle(halo)

    # comparison to reference:
    # tangential shear in x-direction is purely negative in g1
    np.testing.assert_allclose(gamma1, -ref[:,2], rtol=1e-4,
                               err_msg="Computation of shear deviates from reference.")
    np.testing.assert_allclose(gamma2, 0., atol=1e-8,
                               err_msg="Computation of shear deviates from reference.")
    np.testing.assert_allclose(g1, -ref[:,3], rtol=1e-4,
                               err_msg="Computation of reduced shear deviates from reference.")
    np.testing.assert_allclose(g2, 0., atol=1e-8,
                               err_msg="Computation of reduced shear deviates from reference.")
    np.testing.assert_allclose(kappa, ref[:,4], rtol=1e-4,
                               err_msg="Computation of convergence deviates from reference.")

@timer
def test_halo_pos():
    """Test an NFWHalo with a non-zero halo_pos"""

    ref = np.loadtxt(refdir + '/nfw_lens.dat')
    halo = galsim.NFWHalo(mass=1e15, conc=4, redshift=1, halo_pos=galsim.PositionD(2,3))
    pos_x = np.arange(1.,600) + 2     # Adjust x,y values by the same amount.
    pos_y = np.zeros_like(pos_x) + 3
    z_s = np.zeros_like(pos_x) + 2

    check_pickle(halo)

    # comparison to reference should still work.
    kappa = halo.getConvergence((pos_x, pos_y), z_s)
    gamma1, gamma2 = halo.getShear([pos_x, pos_y], z_s, reduced=False)
    g1, g2 = halo.getShear([pos_x, pos_y], z_s)
    mu = halo.getMagnification((pos_x, pos_y), z_s)
    alt_g1, alt_g2, alt_mu = halo.getLensing((pos_x, pos_y), z_s)

    np.testing.assert_allclose(gamma1, -ref[:,2], rtol=1e-4,
                               err_msg="Computation of shear deviates from reference.")
    np.testing.assert_allclose(gamma2, 0., atol=1e-8,
                               err_msg="Computation of shear deviates from reference.")
    np.testing.assert_allclose(g1, -ref[:,3], rtol=1e-4,
                               err_msg="Computation of reduced shear deviates from reference.")
    np.testing.assert_allclose(g2, 0., atol=1e-8,
                               err_msg="Computation of reduced shear deviates from reference.")
    np.testing.assert_allclose(kappa, ref[:,4], rtol=1e-4,
                               err_msg="Computation of convergence deviates from reference.")
    np.testing.assert_array_equal(mu, 1./( (1-kappa)**2 - gamma1**2 - gamma2**2 ),
                                  err_msg="Computation of magnification mu incorrect.")
    np.testing.assert_array_equal(alt_g1, g1,
                                  err_msg="getLensing returned wrong g1")
    np.testing.assert_array_equal(alt_g2, g2,
                                  err_msg="getLensing returned wrong g2")
    np.testing.assert_array_equal(alt_mu, mu,
                                  err_msg="getLensing returned wrong mu")



@timer
def test_cosmology():
    """Test the NFWHalo class in conjunction with non-default cosmologies"""
    # MJ: I don't really have a good way to test that the NFWHalo class is accurate with respect
    # to the cosmology.  If someone with a more theoretical bent is interested in writing some
    # unit tests here, that would be fabulous!
    # All I'm doing here is testing that pickling, repr, etc. work correctly.
    # And the internal consistency checks from above.

    pos_x = np.arange(1,600)
    pos_y = np.zeros_like(pos_x)
    z_s = 2

    for wm, wl in [ (0.4,0.0), (0.3,0.7), (0.25, 0.8) ]:
        cosmo = galsim.Cosmology(omega_m=wm, omega_lam=wl)

        np.testing.assert_equal(cosmo.omega_m, wm)
        np.testing.assert_equal(cosmo.omega_lam, wl)
        np.testing.assert_equal(cosmo.omega_c, 1.-wm-wl)

        halo = galsim.NFWHalo(mass=1e15, conc=4, redshift=1, cosmo=cosmo)
        halo2 = galsim.NFWHalo(mass=1e15, conc=4, redshift=1, omega_m=wm, omega_lam=wl)

        kappa = halo.getConvergence((pos_x, pos_y), z_s)
        gamma1, gamma2 = halo.getShear((pos_x, pos_y), z_s, reduced=False)
        g1, g2 = halo.getShear((pos_x, pos_y), z_s, reduced=True)

        # check internal correctness:
        # g1 = gamma1/(1-kappa), and g2 = 0
        np.testing.assert_allclose(g1, gamma1/(1-np.array(kappa)), rtol=1.e-10,
                                   err_msg="Computation of reduced shear g incorrect.")
        np.testing.assert_array_equal(g2, np.zeros_like(g2),
                                      err_msg="Computation of reduced shear g2 incorrect.")

        check_pickle(cosmo)
        check_pickle(halo)
        check_pickle(halo2)
        assert halo == halo2

        assert_raises(ValueError, cosmo.Da, -0.1)
        assert_raises(ValueError, cosmo.Da, 2.1, 2.3)
        assert_raises(TypeError, galsim.NFWHalo, 1e15, 4, 1, cosmo=5)
        assert_raises(TypeError, galsim.NFWHalo, 1e15, 4, 1, cosmo=cosmo, omega_m=wm)
        assert_raises(TypeError, galsim.NFWHalo, 1e15, 4, 1, cosmo=cosmo, omega_lam=wl)


@timer
def test_shear_variance():
    """Test that shears from several toy power spectra have the expected variances."""
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
    rng2 = rng.duplicate()
    assert_raises(galsim.GalSimError, test_ps.nRandCallsForBuildGrid)
    g1, g2 = test_ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                               units=galsim.degrees)
    assert g1.shape == (ngrid, ngrid)
    assert g2.shape == (ngrid, ngrid)

    # Test nRandCallsForBuildGrid:
    rng2.discard(test_ps.nRandCallsForBuildGrid())
    assert rng == rng2

    # Now we should compare the variance with the predictions.  We use
    # ../devel/modules/lensing_engine.pdf section 5.3 to get
    # Var(g1) + Var(g2) = (1/pi^2) [(pi klim^2 / 4) - kmin^2]
    # Here the 1 on top of the pi^2 is actually P0 which has units of arcsec^2.
    # A final point for this test is that the result should be twice as large than that prediction
    # since we have both E and B power.  And we know from before that due to various effects, the
    # results should actually be ~1.5% too low.
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    predicted_variance *= 2
    var1 = np.var(g1)
    var2 = np.var(g2)
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance from flat power spectrum!"

    # check: are g1, g2 uncorrelated with each other?
    top= np.sum((g1-np.mean(g1))*(g2-np.mean(g2)))
    bottom1 = np.sum((g1-np.mean(g1))**2)
    bottom2 = np.sum((g2-np.mean(g2))**2)
    corr = top / np.sqrt(bottom1*bottom2)
    np.testing.assert_almost_equal(
        corr, 0., decimal=1,
        err_msg="Shear components should be uncorrelated with each other! (flat power spectrum)")

    # Now do the same test as previously, but with E-mode power only.
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim)
    g1, g2 = test_ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                               units='degrees')
    assert g1.shape == (ngrid, ngrid)
    assert g2.shape == (ngrid, ngrid)
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    var1 = np.var(g1)
    var2 = np.var(g2)
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance from flat E-mode power spectrum!"

    # check: are g1, g2 uncorrelated with each other?
    top= np.sum((g1-np.mean(g1))*(g2-np.mean(g2)))
    bottom1 = np.sum((g1-np.mean(g1))**2)
    bottom2 = np.sum((g2-np.mean(g2))**2)
    corr = top / np.sqrt(bottom1*bottom2)
    np.testing.assert_almost_equal(
        corr, 0., decimal=1,
        err_msg="Shear components should be uncorrelated with each other! (flat E-mode power spec.)")

    # check for proper scaling with grid spacing, for fixed number of grid points
    grid_size = 25. # degrees
    ngrid = 500 # grid points
    klim = klim_test
    kmin = 2.*np.pi/grid_size/3600. # arcsec^-1
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim, b_power_function=pk_flat_lim)
    g1, g2 = test_ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                               units=galsim.degrees)
    assert g1.shape == (ngrid, ngrid)
    assert g2.shape == (ngrid, ngrid)
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    predicted_variance *= 2
    var1 = np.var(g1)
    var2 = np.var(g2)
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance from flat power spectrum with smaller grid_size"

    # check for proper scaling with number of grid points, for fixed grid spacing
    grid_size = 25. # degrees
    ngrid = 250 # grid points
    klim = klim_test
    kmin = 2.*np.pi/grid_size/3600. # arcsec^-1
    test_ps = galsim.PowerSpectrum(e_power_function=pk_flat_lim, b_power_function=pk_flat_lim)
    g1, g2 = test_ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                               units=galsim.degrees)
    assert g1.shape == (ngrid, ngrid)
    assert g2.shape == (ngrid, ngrid)
    predicted_variance = (1./np.pi**2)*(0.25*np.pi*(klim**2) - kmin**2)
    predicted_variance *= 2
    var1 = np.var(g1)
    var2 = np.var(g2)
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance from flat power spectrum with smaller ngrid"

    # Test one other theoretical PS: the Gaussian P(k).
    # We define it as P(k) = exp(-s^2 k^2 / 2).
    # First set up the grid.
    grid_size = 50. # degrees
    ngrid = 500 # grid points
    kmin = 2.*np.pi/grid_size/3600.
    kmax = np.pi/(grid_size/ngrid)/3600.
    # Now choose s such that s*kmax=2.5, i.e., very little power at kmax.
    s = 2.5/kmax
    test_ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    g1, g2 = test_ps.buildGrid(grid_spacing = grid_size/ngrid, ngrid=ngrid, rng=rng,
                               units=galsim.degrees)
    assert g1.shape == (ngrid, ngrid)
    assert g2.shape == (ngrid, ngrid)
    # For this case, the prediction for the variance is:
    # Var(g1) + Var(g2) = [1/(2 pi s^2)] * ( (Erf(s*kmax/sqrt(2)))^2 - (Erf(s*kmin/sqrt(2)))^2 )
    try:
        erfmax = math.erf(s*kmax/math.sqrt(2.))
        erfmin = math.erf(s*kmin/math.sqrt(2.))
    except: # For python2.6, which doesn't have math.erf.
        erfmax = 0.9875806693484477
        erfmin = 0.007978712629263206
    var1 = np.var(g1)
    var2 = np.var(g2)
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance from Gaussian power spectrum"

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
    assert g1.shape == (ngrid, ngrid)
    assert g2.shape == (ngrid, ngrid)
    var1 = np.var(g1)
    var2 = np.var(g2)
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance from Gaussian power spectrum with smaller grid_size"

    # check for proper scaling with number of grid points, for fixed grid spacing
    grid_size = 25. # degrees
    ngrid = 250 # grid points
    kmin = 2.*np.pi/grid_size/3600.
    kmax = np.pi/(grid_size/ngrid)/3600.
    # Here one of the Erf[...] values does change.
    try:
        erfmin = math.erf(s*kmin/math.sqrt(2.))
    except:
        erfmin = 0.01595662743380396
    s = 2.5/kmax
    test_ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    g1, g2 = test_ps.buildGrid(grid_spacing = grid_size/ngrid, ngrid=ngrid,
                               rng=rng, units=galsim.degrees)
    assert g1.shape == (ngrid, ngrid)
    assert g2.shape == (ngrid, ngrid)
    var1 = np.var(g1)
    var2 = np.var(g2)
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance from Gaussian power spectrum with smaller ngrid"

    # change grid spacing implicitly via kmax_factor
    # This and the next test can be made at higher precision (0.5% rather than 1.5%), since the
    # grids actually used to make the shears have more points, so they are more accurate.
    grid_size = 50. # degrees
    ngrid = 500 # grid points
    kmax_factor = 2
    kmin = 2.*np.pi/grid_size/3600.
    kmax = np.pi/(grid_size/ngrid)/3600.*kmax_factor
    try:
        erfmin = math.erf(s*kmin/math.sqrt(2.))
    except:
        erfmin = 0.007978712629263206
    s = 2.5/kmax
    test_ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    rng2 = rng.duplicate()
    g1, g2 = test_ps.buildGrid(grid_spacing = grid_size/ngrid, ngrid=ngrid,
                               rng=rng, units=galsim.degrees, kmax_factor=kmax_factor)
    assert g1.shape == (ngrid, ngrid)
    assert g2.shape == (ngrid, ngrid)
    var1 = np.var(g1)
    var2 = np.var(g2)
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance from Gaussian power spectrum with kmax_factor=2"
    rng2.discard(test_ps.nRandCallsForBuildGrid())
    assert rng == rng2

    # change ngrid implicitly with kmin_factor
    grid_size = 50. # degrees
    ngrid = 500 # grid points
    kmin_factor = 2
    kmin = 2.*np.pi/grid_size/3600./kmin_factor
    kmax = np.pi/(grid_size/ngrid)/3600.
    s = 2.5/kmax
    # This time, erfmin is smaller.
    try:
        erfmin = math.erf(s*kmin/math.sqrt(2.))
    except:
        erfmin = 0.003989406181481644
    # Also, for this test, it should be equivalent to use the b-mode instead.
    test_ps = galsim.PowerSpectrum(b_power_function=lambda k : np.exp(-0.5*((s*k)**2)))
    g1, g2 = test_ps.buildGrid(grid_spacing = grid_size/ngrid, ngrid=ngrid,
                               rng=rng, units=galsim.degrees, kmin_factor=kmin_factor)
    assert g1.shape == (ngrid, ngrid)
    assert g2.shape == (ngrid, ngrid)
    var1 = np.var(g1)
    var2 = np.var(g2)
    predicted_variance = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance from Gaussian power spectrum with kmin_factor=2"
    rng2.discard(test_ps.nRandCallsForBuildGrid())
    assert rng == rng2

    # Now check the variances post-interpolation to random (off-grid) points.  Ideally, our default
    # interpolant should not alter the power spectrum very much from kmin to kmax, so the shear
    # variance should also not be significantly altered.  To test this, we take the g1, g2 from the
    # previous buildGrid() call, interpolate to some random positions that are not too near the
    # edges (since near the edges there are known artifacts), and check the variances.
    grid_spacing = grid_size/ngrid
    min = (-ngrid/2 + 0.5) * grid_spacing
    max = (ngrid/2 + 0.5) * grid_spacing
    # Now chop the outer ~25% off just to be conservative.  Since min and max are negative and
    # positive, respectively, we'll just multiply them by 0.75 to make the grid smaller.
    min *= 0.75
    max *= 0.75
    # Generate a bunch of random points:
    n_rand = 10000
    x = np.zeros(n_rand)
    y = np.zeros(n_rand)
    ud = galsim.UniformDeviate(12345)
    for i in range(n_rand):
        x[i] = min + (max-min)*ud()
        y[i] = min + (max-min)*ud()
    # Get the shears at those points
    g1, g2 = test_ps.getShear(pos=(x,y), units=galsim.degrees, reduced=False)
    var1 = np.var(g1)
    var2 = np.var(g2)
    # Use the predicted variance from before
    print('predicted variance = ',predicted_variance)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/predicted_variance-1))
    assert np.abs((var1+var2) - predicted_variance) < tolerance_var * predicted_variance, \
            "Incorrect shear variance post-interpolation"

    # Warn for accessing values outside of valid bounds (and not periodic)
    assert_warns(galsim.GalSimWarning, test_ps.getShear, pos=(max*2, 0), units='deg')
    assert_warns(galsim.GalSimWarning, test_ps.getShear, pos=(max*2, 0), reduced=False, units='deg')
    assert_warns(galsim.GalSimWarning, test_ps.getConvergence, pos=(max*2, 0), units='deg')
    assert_warns(galsim.GalSimWarning, test_ps.getMagnification, pos=(max*2, 0), units='deg')
    assert_warns(galsim.GalSimWarning, test_ps.getLensing, pos=(max*2, 0), units='deg')


@timer
def test_shear_seeds():
    """Test that shears from lensing engine behave appropriate when given same/different seeds"""
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


@timer
def test_shear_reference():
    """Test shears from lensing engine compared to stored reference values"""
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
    g1, g2, kappa = ps.buildGrid(grid_spacing = dx, ngrid = n, rng=rng, get_convergence=True,
                                 bandlimit = None) # Switch off this default, since original set of
                                 # shears were computed before the bandlimit option existed.

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


@timer
def test_delta2():
    """Test that using delta2 gives appropriate equivalent power spectrum. """

    rng = galsim.BaseDeviate(512342)
    grid_size = 10. # degrees
    ngrid = 100 # grid points
    grid_spacing = grid_size / ngrid

    for bandlimit in [None, 'soft', 'hard']:
        for func in [pk2, pk1, pk_flat_lim]:
            dfunc = lambda k: k**2 * func(k) / (2.*np.pi)

            # E only
            ps_ref = galsim.PowerSpectrum(e_power_function=func, units='deg')
            g1_ref, g2_ref = ps_ref.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid,
                                              rng=rng.duplicate(), units='deg')
            ps_delta = galsim.PowerSpectrum(e_power_function=dfunc, units='deg', delta2=True)
            g1_delta, g2_delta = ps_delta.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid,
                                                    rng=rng.duplicate(), units='deg')
            np.testing.assert_allclose(g1_delta, g1_ref, rtol=1.e-8)
            np.testing.assert_allclose(g2_delta, g2_ref, rtol=1.e-8)

            # B only
            ps_ref = galsim.PowerSpectrum(b_power_function=func, units='deg')
            g1_ref, g2_ref = ps_ref.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid,
                                              rng=rng.duplicate(), units='deg')
            ps_delta = galsim.PowerSpectrum(b_power_function=dfunc, units='deg', delta2=True)
            g1_delta, g2_delta = ps_delta.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid,
                                                    rng=rng.duplicate(), units='deg')
            np.testing.assert_allclose(g1_delta, g1_ref, rtol=1.e-8)
            np.testing.assert_allclose(g2_delta, g2_ref, rtol=1.e-8)

            # E and B
            ps_ref = galsim.PowerSpectrum(e_power_function=func, b_power_function=func,
                                          units='deg')
            g1_ref, g2_ref = ps_ref.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid,
                                              rng=rng.duplicate(), units='deg')
            ps_delta = galsim.PowerSpectrum(e_power_function=dfunc, b_power_function=dfunc,
                                            units='deg', delta2=True)
            g1_delta, g2_delta = ps_delta.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid,
                                                    rng=rng.duplicate(), units='deg')
            np.testing.assert_allclose(g1_delta, g1_ref, rtol=1.e-8)
            np.testing.assert_allclose(g2_delta, g2_ref, rtol=1.e-8)

    # The above ps isn't picklable, since we use a lambda function for dfunc.
    # So give it a string to test the pickling and repr with delta2=True
    ps_delta = galsim.PowerSpectrum(e_power_function='k**3 / 2*np.pi', units='deg', delta2=True)
    check_pickle(ps_delta)


@timer
def test_shear_get():
    """Check that using gridded outputs and the various getFoo methods gives consistent results"""
    # choose a power spectrum and grid setup
    my_ps = galsim.PowerSpectrum(lambda k : k**0.5)
    # build the grid
    grid_spacing = 17.
    ngrid = 100

    # Before calling buildGrid, these are invalid
    assert_raises(galsim.GalSimError, my_ps.getShear, galsim.PositionD(0,0))
    assert_raises(galsim.GalSimError, my_ps.getConvergence, galsim.PositionD(0,0))
    assert_raises(galsim.GalSimError, my_ps.getMagnification, galsim.PositionD(0,0))
    assert_raises(galsim.GalSimError, my_ps.getLensing, galsim.PositionD(0,0))

    g1, g2, kappa = my_ps.buildGrid(grid_spacing = grid_spacing, ngrid = ngrid,
                                    get_convergence = True)
    min = (-ngrid/2 + 0.5) * grid_spacing
    max = (ngrid/2 - 0.5) * grid_spacing
    x, y = np.meshgrid(np.arange(min,max+grid_spacing,grid_spacing),
                       np.arange(min,max+grid_spacing,grid_spacing))

    # convert theoretical to observed quantities for grid
    g1_r, g2_r, mu = galsim.lensing_ps.theoryToObserved(g1, g2, kappa)

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

    # Test single position versions
    np.testing.assert_almost_equal(my_ps.getShear((x[0,0], y[0,0])), (g1_r[0,0], g2_r[0,0]))
    np.testing.assert_almost_equal(my_ps.getShear((x[0,0], y[0,0]), reduced=False),
                                   (g1[0,0], g2[0,0]))
    np.testing.assert_almost_equal(my_ps.getConvergence((x[0,0], y[0,0])), kappa[0,0])
    np.testing.assert_almost_equal(my_ps.getMagnification((x[0,0], y[0,0])), mu[0,0])
    np.testing.assert_almost_equal(my_ps.getLensing((x[0,0], y[0,0])),
                                   (g1_r[0,0], g2_r[0,0], mu[0,0]))

    # Test outside of bounds
    with assert_warns(galsim.GalSimWarning):
        np.testing.assert_almost_equal(my_ps.getShear((5000,5000)), (0,0))
        np.testing.assert_almost_equal(my_ps.getShear((5000,5000), reduced=False), (0,0))
        np.testing.assert_almost_equal(my_ps.getConvergence((5000,5000)), 0)
        np.testing.assert_almost_equal(my_ps.getMagnification((5000,5000)), 1)
        np.testing.assert_almost_equal(my_ps.getLensing((5000,5000)), (0,0,1))



@timer
def test_shear_units():
    """Test that the shears we get out do not depend on the input PS and grid units."""
    rand_seed = 123456

    grid_size = 10. # degrees
    ngrid = 100
    grid_spacing = 3600. * grid_size / ngrid

    # Define a PS with some normalization value P(k=1/arcsec)=1 arcsec^2.
    # For this case we are getting the shears using units of arcsec for everything.
    ps = galsim.PowerSpectrum(lambda k : k)
    g1, g2 = ps.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid,
                          rng = galsim.BaseDeviate(rand_seed))
    # The above was done with all inputs given in arcsec.  Now, redo it, inputting the PS
    # information in degrees and the grid info in arcsec.
    # We know that if k=1/arcsec, then when expressed as 1/degrees, it is
    # k=3600/degree.  So define the PS as P(k=3600/degree)=(1/3600.)^2 degree^2.
    ps = galsim.PowerSpectrum(lambda k : (1./3600.**2)*(k/3600.), units=galsim.degrees)
    g1_2, g2_2 = ps.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid,
                              rng=galsim.BaseDeviate(rand_seed))
    # Finally redo it, inputting the PS and grid info in degrees.
    ps = galsim.PowerSpectrum(lambda k : (1./3600.**2)*(k/3600.), units='degrees')
    g1_3, g2_3 = ps.buildGrid(grid_spacing=grid_spacing/3600., ngrid=ngrid,
                              units='degrees', rng=galsim.BaseDeviate(rand_seed))

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

    # Can also change the units in the getShear function
    origin = galsim.PositionD(-grid_size/2. * 3600. + grid_spacing/2.,
                              -grid_size/2. * 3600. + grid_spacing/2.)
    g1_4, g2_4 = ps.getShear(origin, reduced=False)
    np.testing.assert_almost_equal(g1_4, g1[0,0], decimal=12)
    np.testing.assert_almost_equal(g2_4, g2[0,0], decimal=12)

    origin /= 3600.
    g1_5, g2_5 = ps.getShear(origin, reduced=False, units=galsim.degrees)
    np.testing.assert_almost_equal(g1_5, g1[0,0], decimal=12)
    np.testing.assert_almost_equal(g2_5, g2[0,0], decimal=12)

    origin *= 60.
    g1_6, g2_6 = ps.getShear(origin, reduced=False, units='arcmin')
    np.testing.assert_almost_equal(g1_6, g1[0,0], decimal=12)
    np.testing.assert_almost_equal(g2_6, g2[0,0], decimal=12)

    # Check ne
    ps = galsim.PowerSpectrum('k', 'k**2', False, 'arcsec')
    assert ps == galsim.PowerSpectrum(e_power_function='k', b_power_function='k**2')
    assert ps == galsim.PowerSpectrum(e_power_function='k', b_power_function='k**2',
                                      delta2=False, units=galsim.arcsec)
    diff_ps_list = [ps,
                    galsim.PowerSpectrum('k**2', 'k**2', False, 'arcsec'),
                    galsim.PowerSpectrum('k', 'k', False, 'arcsec'),
                    galsim.PowerSpectrum('k', 'k**2', True, 'arcsec'),
                    galsim.PowerSpectrum('k', 'k**2', False, 'arcmin')]
    check_all_diff(diff_ps_list)

@timer
def test_tabulated():
    """Test using a LookupTable to interpolate a P(k) that is known at certain k"""
    # make PowerSpectrum with some obvious analytic form, P(k)=k^2
    ps_analytic = galsim.PowerSpectrum(pk2)

    # now tabulate that analytic form at a range of k
    k_arr = 0.01*np.arange(10000.)+0.01
    p_arr = k_arr**(2.)

    # make a LookupTable to initialize another PowerSpectrum
    tab = galsim.LookupTable(k_arr, p_arr)
    ps_tab = galsim.PowerSpectrum(tab)
    check_pickle(ps_tab)  # This is the first one that doesn't use a function, so it is picklable.

    # draw shears on a grid from both PowerSpectrum objects, with same random seed
    seed = 12345
    g1_analytic, g2_analytic = ps_analytic.buildGrid(grid_spacing = 1.7, ngrid = 10,
                                                     rng = galsim.BaseDeviate(seed))
    g1_tab, g2_tab = ps_tab.buildGrid(grid_spacing = 1.7, ngrid = 10,
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
    ps_tab2 = galsim.PowerSpectrum(filename)
    check_pickle(ps_tab2)
    g1_tab2, g2_tab2 = ps_tab2.buildGrid(grid_spacing = 1.7, ngrid = 10,
                                         rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab2, 6,
        err_msg = "g1 from file-based tabulated P(k) differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab2, 6,
        err_msg = "g2 from file-based tabulated P(k) differs from expectation!")
    # check that we get the same answer whether we use interpolation in log for k, P, or both
    tab = galsim.LookupTable(k_arr, p_arr, x_log = True)
    ps_tab = galsim.PowerSpectrum(tab)
    check_pickle(ps_tab2)
    g1_tab, g2_tab = ps_tab.buildGrid(grid_spacing = 1.7, ngrid = 10,
                                      rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg = "g1 of shear field from tabulated P(k) with x_log differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg = "g2 of shear field from tabulated P(k) with x_log differs from expectation!")
    tab = galsim.LookupTable(k_arr, p_arr, f_log = True)
    ps_tab = galsim.PowerSpectrum(tab)
    check_pickle(ps_tab)
    g1_tab, g2_tab = ps_tab.buildGrid(grid_spacing = 1.7, ngrid = 10,
                                      rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg = "g1 of shear field from tabulated P(k) with f_log differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg = "g2 of shear field from tabulated P(k) with f_log differs from expectation!")
    tab = galsim.LookupTable(k_arr, p_arr, x_log = True, f_log = True)
    ps_tab = galsim.PowerSpectrum(tab)
    check_pickle(ps_tab)
    g1_tab, g2_tab = ps_tab.buildGrid(grid_spacing = 1.7, ngrid = 10,
                                      rng = galsim.BaseDeviate(seed))
    np.testing.assert_almost_equal(g1_analytic, g1_tab, 6,
        err_msg="g1 of shear field from tabulated P(k) with x_log, f_log differs from expectation!")
    np.testing.assert_almost_equal(g2_analytic, g2_tab, 6,
        err_msg="g2 of shear field from tabulated P(k) with x_log, f_log differs from expectation!")

    # check for appropriate response to inputs when making/using LookupTable
    ## mistaken interpolant choice
    assert_raises(ValueError, galsim.LookupTable, k_arr, p_arr, interpolant='splin')
    ## k, P arrays not the same size
    assert_raises(ValueError, galsim.LookupTable, 0.01*np.arange(100.), p_arr)
    ## try to make shears, but grid includes k values that were not part of the originally
    ## tabulated P(k) (for this test we make a stupidly limited k grid just to ensure that an
    ## exception should be raised)
    t = galsim.LookupTable((0.99,1.,1.01),(0.99,1.,1.01))
    limited_ps = galsim.PowerSpectrum(t)
    check_pickle(limited_ps)
    assert_raises(ValueError, limited_ps.buildGrid, grid_spacing=1.7, ngrid=100)

    ## try to interpolate in log, but with zero values included
    assert_raises(ValueError, galsim.LookupTable, (0.,1.,2.), (0.,1.,2.), x_log=True)
    assert_raises(ValueError, galsim.LookupTable, (0.,1.,2.), (0.,1.,2.), f_log=True)
    assert_raises(ValueError, galsim.LookupTable, (0.,1.,2.), (0.,1.,2.), x_log=True, f_log=True)

    # Negative power is invalid.
    neg_power = galsim.LookupTable(k_arr, np.cos(k_arr))
    print('neg_power = ',neg_power)
    with assert_raises(galsim.GalSimError):
        negps = galsim.PowerSpectrum(neg_power)
        negps.buildGrid(grid_spacing=1.7, ngrid=10)

    # Check some invalid PowerSpectrum parameters
    assert_raises(ValueError, galsim.PowerSpectrum)
    assert_raises(ValueError, galsim.PowerSpectrum, delta2=True)
    assert_raises(ValueError, galsim.PowerSpectrum, delta2=True, units='radians')
    assert_raises(ValueError, galsim.PowerSpectrum, e_power_function=tab, units='inches')
    assert_raises(ValueError, galsim.PowerSpectrum, e_power_function=tab, units=True)
    assert_raises(ValueError, galsim.PowerSpectrum, e_power_function='not_a_file')
    assert_raises(ValueError, galsim.PowerSpectrum, b_power_function='not_a_file')
    assert_raises(TypeError, ps_tab.buildGrid)
    assert_raises(TypeError, ps_tab.buildGrid, grid_spacing=1.7)
    assert_raises(TypeError, ps_tab.buildGrid, ngrid=10)
    assert_raises(ValueError, ps_tab.buildGrid, grid_spacing=1.7, ngrid=10.5)
    assert_raises(ValueError, ps_tab.buildGrid, grid_spacing=1.7, ngrid=10, kmin_factor=2.5)
    assert_raises(ValueError, ps_tab.buildGrid, grid_spacing=1.7, ngrid=10, kmax_factor=1.5)
    assert_raises(ValueError, ps_tab.buildGrid, grid_spacing=1.7, ngrid=10, center=(5,5))
    assert_raises(ValueError, ps_tab.buildGrid, grid_spacing=1.7, ngrid=10, units='inches')
    assert_raises(ValueError, ps_tab.buildGrid, grid_spacing=1.7, ngrid=10, units=True)
    assert_raises(ValueError, ps_tab.buildGrid, grid_spacing=1.7, ngrid=10, bandlimit='none')
    assert_raises(TypeError, ps_tab.getShear)
    assert_raises(TypeError, ps_tab.getShear, pos=())
    assert_raises(TypeError, ps_tab.getShear, pos=3)
    assert_raises(TypeError, ps_tab.getShear, pos=(3,))
    assert_raises(TypeError, ps_tab.getShear, pos=(3,4,5))
    assert_raises(ValueError, ps_tab.getShear, pos=(3,4), units='invalid')
    assert_raises(ValueError, ps_tab.getShear, pos=(3,4), units=17)

    # check that when calling LookupTable, you can provide a scalar, list, tuple or array
    tab = galsim.LookupTable(k_arr, p_arr)
    k = 0.5
    assert type(tab(k)) == float
    k = (0.5, 1.5)
    result = tab(k)
    assert len(result) == len(k)
    k = list(k)
    result = tab(k)
    assert len(result) == len(k)
    k = np.array(k)
    result = tab(k)
    assert len(result) == len(k)
    k = 0.01+np.zeros((2,2))
    result = tab(k)
    assert result.shape == k.shape

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


@timer
def test_kappa_gauss():
    """Test that our Kaiser-Squires inversion routine correctly recovers the convergence map
    for a field containing known Gaussian-profile halos.
    """
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
    k_testE, k_testB = galsim.lensing_ps.kappaKaiserSquires(g1, g2)
    # Then run tests based on the central region to avoid edge effects (known issue with KS
    # inversion)
    icent = np.arange(ngrid // 2) + ngrid // 4
    # Test that E-mode kappa matches
    np.testing.assert_array_almost_equal(
        k_testE[np.ix_(icent, icent)], k_ref[np.ix_(icent, icent)], decimal=2,
        err_msg="Reconstructed kappa does not match input to 2 decimal places.")
    # Test B-mode kappa is consistent with zero
    np.testing.assert_array_almost_equal(
        k_testB[np.ix_(icent, icent)], np.zeros((ngrid // 2, ngrid // 2)), decimal=3,
        err_msg="Reconstructed B-mode kappa is non-zero at greater than 3 decimal places.")
    # Generate shears using the 45 degree rotation option
    g1r_big, g2r_big = shear_gaussian(
        x, y, sigma=5., pos=galsim.PositionD(-6., -6.), amp=.5, rotate45=True)
    g1r_sml, g2r_sml = shear_gaussian(
        x, y, sigma=4., pos=galsim.PositionD(6., 6.), amp=.2, rotate45=True)
    g1r = g1r_big + g1r_sml
    g2r = g2r_big + g2r_sml
    kr_testE, kr_testB = galsim.lensing_ps.kappaKaiserSquires(g1r, g2r)
    # Test that B-mode kappa for rotated shear field matches E mode
    np.testing.assert_array_almost_equal(
        kr_testB[np.ix_(icent, icent)], k_ref[np.ix_(icent, icent)], decimal=2,
        err_msg="Reconstructed kappaB in rotated shear field does not match original kappaE to 2 "+
        "decimal places.")
    # Test E-mode kappa is consistent with zero for rotated shear field
    np.testing.assert_array_almost_equal(
        kr_testE[np.ix_(icent, icent)], np.zeros((ngrid // 2, ngrid // 2)), decimal=3,
        err_msg="Reconstructed kappaE is non-zero at greater than 3 decimal places for rotated "+
        "shear field.")

    assert_raises(TypeError, galsim.lensing_ps.kappaKaiserSquires, g1=0.3, g2=0.1)
    assert_raises(ValueError, galsim.lensing_ps.kappaKaiserSquires, g1=g1, g2=g2[:50,:50])
    assert_raises(NotImplementedError, galsim.lensing_ps.kappaKaiserSquires,
                  g1=g1[:,:50], g2=g2[:,:50])


@timer
def test_power_spectrum_with_kappa():
    """Test that the convergence map generated by the PowerSpectrum class is consistent with the
    Kaiser Squires inversion of the corresponding shear field.
    """
    # Note that in order for this test to pass, we have to control aliasing by smoothing the power
    # spectrum to go to zero above some maximum k.  This is the only way to get agreement at high
    # precision between the gamma's and kappa's from the lensing engine vs. that from a Kaiser and
    # Squires inversion.
    rseed=177774
    ngrid=100
    dx_grid_arcmin = 6
    # First lookup a cosmologically relevant power spectrum (bandlimited version to remove aliasing
    # and allow high-precision comparison).
    tab_ps = galsim.LookupTable.from_file('../examples/data/cosmo-fid.zmed1.00_smoothed.out')

    # Begin with E-mode input power
    psE = galsim.PowerSpectrum(tab_ps, None, units=galsim.radians)
    check_pickle(psE)
    g1E, g2E, k_test = psE.buildGrid(
        grid_spacing=dx_grid_arcmin, ngrid=ngrid, units='arcmin',
        rng=galsim.BaseDeviate(rseed), get_convergence=True)
    kE_ks, kB_ks = galsim.lensing_ps.kappaKaiserSquires(g1E, g2E)
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
    check_pickle(psB)
    g1B, g2B, k_test = psB.buildGrid(
        grid_spacing=dx_grid_arcmin, ngrid=ngrid, units=galsim.arcmin,
        rng=galsim.BaseDeviate(rseed), get_convergence=True)
    kE_ks, kB_ks = galsim.lensing_ps.kappaKaiserSquires(g1B, g2B)
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
    kE_ks_rotated, kB_ks_rotated = galsim.lensing_ps.kappaKaiserSquires(g2B, -g1B)
    np.testing.assert_array_almost_equal(
        kE_ks_rotated, kB_ks, decimal=exact_dp,
        err_msg="KS inverted kappaE from B-mode only PowerSpectrum fails rotation test.")
    np.testing.assert_array_almost_equal(
        kB_ks_rotated, np.zeros_like(kB_ks), decimal=exact_dp,
        err_msg="KS inverted kappaB from B-mode only PowerSpectrum fails rotation test.")

    # Finally, do E- and B-mode power
    psB = galsim.PowerSpectrum(tab_ps, tab_ps, units=galsim.radians)
    check_pickle(psB)
    g1EB, g2EB, k_test = psB.buildGrid(
        grid_spacing=dx_grid_arcmin, ngrid=ngrid, units=galsim.arcmin,
        rng=galsim.BaseDeviate(rseed), get_convergence=True)
    kE_ks, kB_ks = galsim.lensing_ps.kappaKaiserSquires(g1EB, g2EB)
    # Test that E-mode kappa matches to some sensible accuracy
    np.testing.assert_array_almost_equal(
        k_test, kE_ks, decimal=exact_dp,
        err_msg="E/B PowerSpectrum output kappa does not match KS inversion to 16 d.p.")
    # Test rotating the shears by 45 degrees
    kE_ks_rotated, kB_ks_rotated = galsim.lensing_ps.kappaKaiserSquires(g2EB, -g1EB)
    np.testing.assert_array_almost_equal(
        kE_ks_rotated, kB_ks, decimal=exact_dp,
        err_msg="KS inverted kappaE from E/B PowerSpectrum fails rotation test.")
    np.testing.assert_array_almost_equal(
        kB_ks_rotated, -kE_ks, decimal=exact_dp,
        err_msg="KS inverted kappaB from E/B PowerSpectrum fails rotation test.")


@timer
def test_corr_func():
    """Test that the correlation function calculation in calculateXi() works properly.
    """
    # We want to compare the integration that is done by galsim.PowerSpectrum.calculateXi() is
    # accurate.  Ideally this would be done by comparison with some power spectrum for which the
    # conversion to a correlation function is analytic.  Given that the conversion to correlation
    # function involves integration by a Bessel function, there are not many options for
    # power spectra for which there is a closed-form expression for the integral.  For this test I
    # will use the following relations:
    #   \int x^n J_{n-1}(x) dx = x^n J_n(x) + C
    #   \int x^(-n) J_{n+1}(x) dx = -x^{-n} J_n(x) + C
    # The first one is helpful for the case of n=1, i.e.,
    #   \int x J_0(x) dx = x J_1(x) + C
    # which provides a test of xi_+.
    # The second expression is helpful if we use n=3, i.e.,
    #   \int x^(-3) J_4(x) dx = -x^{-3} J_3(x) + C
    # which provides a test of xi_-.
    #
    # Testing with these functions means that our shear power spectra are not cosmological-looking,
    # but I couldn't find an analytic expression that involved something that looked like a real
    # shear power spectrum, so I think we're stuck with these.  At least they are non-trivially
    # interesting/challenging tests.

    # First we will test xi+ calculations.  So, to put the equations above into the proper form,
    # we should keep in mind that what gets returned from the lensing engine is
    #   xi+(r) = (1/2pi) \int_{kmin}^{kmax} P(k) J_0(kr) k dk
    # Let's substitute x = kr to get the integral into a format that can be compared with the
    # expression above.
    #   xi+(r) = (1/2pi) (1/r)^2 \int_{r*kmin}^{r*kmax} P(x) J_0(x) x dx
    # We want the integrand to be x J_0(x), which leads me to conclude that we should set
    # P(k)=1.  In that case we should find
    #   xi+(r) = (1/2pi) (1/r)^2 \int_{r*kmin}^{r*kmax} x J_0(x) dx
    #          = (1/2pi) (1/r)^2 [r kmax J_1(r*kmax) - r kmin J_1(r*kmin)]
    #          = (1/2pi) (1/r) [kmax J_1(r*kmax) - kmin J_1(r*kmin)]
    # Let's only check this at 10 values of theta so the test isn't painfully slow.
    n_theta = 10
    # Also, we're going to just work in arcsec, which is the natural set of units for the
    # lensing engine.  Other unit tests already ensure that the units are working out properly
    # so we will not test that here.
    ps = galsim.PowerSpectrum(lambda k : 1.)
    # Set up a grid, with the expectation that we'll use kmin_factor=kmax_factor=1 in our test:
    ngrid = 100
    grid_spacing = 360. # arcsec, i.e., 0.1 degrees
    # Get test values for xi+; ignore xi- since we don't have an analytic expression for it:
    t, test_xip, _ = ps.calculateXi(grid_spacing=grid_spacing, ngrid=ngrid, n_theta=n_theta,
                                    bandlimit='hard')
    # Now we have to calculate the theoretical values.  First, we need kmin and kmax in
    # 1/arcsec:
    kmin = 2.*np.pi/(ngrid*grid_spacing)
    kmax = np.pi/grid_spacing
    theory_xip = np.zeros_like(t)
    for ind in range(len(theory_xip)):
        theory_xip[ind] = kmax*galsim.bessel.j1(t[ind]*kmax) - kmin*galsim.bessel.j1(t[ind]*kmin)
    theory_xip /= (2.*np.pi*t)
    # Finally, make sure they are equal to 10^{-5}
    np.testing.assert_allclose(test_xip, theory_xip, rtol=1.e-5,
                               err_msg='Integrated xi+ differs from reference values')

    # Repeat with different units
    t, test_xip2, _ = ps.calculateXi(grid_spacing=grid_spacing/3600, ngrid=ngrid, n_theta=n_theta,
                                     bandlimit='hard', units='deg')
    np.testing.assert_array_almost_equal(test_xip2, test_xip, decimal=12)

    # Now, do the test for xi-.  We again have to rearrange equations, starting with the lensing
    # engine output:
    #    xi-(r) = (1/2pi) \int_{kmin}^{kmax} P(k) J_4(kr) k dk
    # Substituting x = kr,
    #    xi-(r) = (1/2pi) (1/r)^2 \int_{r*kmin}^{r*kmax} P(x) J_4(x) x dx
    # We want the integrand to be x^{-3} J_4(x), which suggests P(x) = x^{-4}.
    # But we have to tell the lensing engine P(k), not P(kr).  So we'll tell it that P(k)=k^{-4}
    # and we'll put the r^{-4} part into the result ourselves.
    # In other words, our theory calculation will be:
    #    xi-(r) = (1/2pi) (1/r)^2 [-(r*kmax)^{-3} J_3(r*kmax) + (r*kmin)^{-3} J_3(r*kmin)]
    #           = [kmin^{-3} J_3(r*kmin) - kmax^{-3} J_3(r*kmax)] / (2pi * r^5)
    # and we will compare it with
    #    (lensing engine output for xi-)/r^4
    #
    # Alternatively and more cleanly, we can compare the lensing engine output for xi- with
    # the theory prediction for r^4 xi-(r), which is
    #    [kmin^{-3} J_3(r*kmin) - kmax^{-3} J_3(r*kmax)] / (2pi * r)
    # We begin by slightly fudging the power function to avoid a RuntimeWarning for division by
    # zero: (k+1e-12)^{-4} instead of k^{-4}
    ps = galsim.PowerSpectrum(lambda k : (k+1.e-12)**(-4))
    t, _, test_xim = ps.calculateXi(grid_spacing=grid_spacing, ngrid=ngrid, n_theta=n_theta,
                                           bandlimit='hard')
    # Now we have to calculate the theoretical values.
    theory_xim = np.zeros_like(t)
    for ind in range(len(theory_xim)):
        theory_xim[ind] = (galsim.bessel.jn(3,t[ind]*kmin)/kmin**3 -
                           galsim.bessel.jn(3,t[ind]*kmax)/kmax**3)
    theory_xim /= (2.*np.pi*t)
    # Finally, make sure they are equal to 10^{-5}
    np.testing.assert_allclose(test_xim, theory_xim, rtol=1.e-5,
                               err_msg='Integrated xi- differs from reference values')

    # Test for invalid inputs
    assert_raises(ValueError, ps.calculateXi, grid_spacing='foo', ngrid=10)
    assert_raises(ValueError, ps.calculateXi, grid_spacing, ngrid='bar')
    assert_raises(ValueError, ps.calculateXi, grid_spacing, ngrid, units='gradians')
    assert_raises(ValueError, ps.calculateXi, grid_spacing, ngrid, kmin_factor='1.5')
    assert_raises(ValueError, ps.calculateXi, grid_spacing, ngrid, kmax_factor='1.5')
    assert_raises(ValueError, ps.calculateXi, grid_spacing, ngrid, n_theta='1.5')
    assert_raises(ValueError, ps.calculateXi, grid_spacing, ngrid, bandlimit='none')

    # Test B-mode version
    ps_b = galsim.PowerSpectrum(b_power_function=lambda k : 1.)
    t, b_xip, _ = ps_b.calculateXi(grid_spacing, ngrid, n_theta=n_theta)
    np.testing.assert_allclose(b_xip, theory_xip, rtol=1.e-5,
                               err_msg='B-mode xi+ differs from reference values')
    ps_b = galsim.PowerSpectrum(b_power_function=lambda k : (k+1.e-12)**(-4))
    t, _, b_xim = ps_b.calculateXi(grid_spacing=grid_spacing, ngrid=ngrid, n_theta=n_theta,
                                   bandlimit='hard')
    np.testing.assert_allclose(b_xim, -theory_xim, rtol=1.e-5,
                               err_msg='B-mode xi- differs from reference values')

    # Test E and B
    ps_eb = galsim.PowerSpectrum(e_power_function=lambda k: 1., b_power_function=lambda k : 1.)
    t, eb_xip, _ = ps_eb.calculateXi(grid_spacing, ngrid, n_theta=n_theta)
    np.testing.assert_allclose(eb_xip, 2*theory_xip, rtol=1.e-5,
                               err_msg='E+B xi+ differs from reference values')
    ps_eb = galsim.PowerSpectrum(e_power_function=lambda k : (k+1.e-12)**(-4),
                                b_power_function=lambda k : (k+1.e-12)**(-4))
    t, _, eb_xim = ps_eb.calculateXi(grid_spacing=grid_spacing, ngrid=ngrid, n_theta=n_theta,
                                     bandlimit='hard')
    np.testing.assert_allclose(eb_xim/theory_xim, 0., atol=1.e-5,
                               err_msg='E+B xi- differs from reference values')

    # Repeat with different units
    t, _, test_xim2 = ps.calculateXi(grid_spacing=grid_spacing/3600, ngrid=ngrid, n_theta=n_theta,
                                     bandlimit='hard', units='deg')
    np.testing.assert_array_almost_equal(test_xim2, test_xim, decimal=12)



@timer
def test_periodic():
    """Test that the periodic interpolation option is working properly.
    """
    # Periodic interpolation is an option in the lensing power spectrum module primarily because,
    # with our shear grids being implicitly periodic, it will give the right shear power spectrum
    # within kmin<k<kmax if we do interpolation in some periodic way.
    #
    # We will test this functionality by generating shear on a grid, then using periodic
    # interpolation with the nearest-neighbor interpolant for some grid that has the same ngrid and
    # spacing but some large offset from the original grid coordinates.  The lensing engine should
    # tile the sky with periodic grids, and if we use NN interpolation then we should just get a
    # (wrapped/shifted) copy of the original grid.  Thus the shear power spectrum should be
    # precisely preserved by this operation.

    # Set up a cosmological shear power spectrum.
    tab_ps = galsim.LookupTable.from_file('../examples/data/cosmo-fid.zmed1.00_smoothed.out')
    ps = galsim.PowerSpectrum(tab_ps, units=galsim.radians)
    check_pickle(ps)

    # Set up a grid.  Make it GREAT10/GREAT3-like.
    ngrid = 100
    grid_spacing = 0.1 # degrees

    # Make shears on the grid.
    # Also use non-int (but still integral) values of kmin_factor, kmax_factor to test conversion.
    g1, g2, kappa = ps.buildGrid(ngrid=100, grid_spacing=0.1, units=galsim.degrees,
                                 rng=galsim.UniformDeviate(314159), interpolant='nearest',
                                 kmin_factor=3., kmax_factor=1., get_convergence=True)
    g1_r, g2_r, mu = galsim.lensing_ps.theoryToObserved(g1, g2, kappa)

    # Set up a new set of x, y.  Make a grid and then shift it coherently:
    min = (-ngrid/2 + 0.5) * grid_spacing
    max = (ngrid/2 - 0.5) * grid_spacing
    x, y = np.meshgrid(np.arange(min,max+grid_spacing,grid_spacing),
                       np.arange(min,max+grid_spacing,grid_spacing))
    x += 17.40 # degrees
    y -= 0.617 # degrees

    # Get shears at those positions using periodic interpolation.
    g1_shift, g2_shift = ps.getShear(pos=(x.flatten(),y.flatten()), units=galsim.degrees,
                                     reduced=False, periodic=True)
    g1_shift = g1_shift.reshape((ngrid,ngrid))
    g2_shift = g2_shift.reshape((ngrid,ngrid))
    # Compute shear power spectra for the original grid and the new grid.  We can use all the
    # default settings for the power spectrum estimator.
    pse = galsim.pse.PowerSpectrumEstimator()
    check_pickle(pse)
    k, pe, pb, peb = pse.estimate(g1, g2)
    _, pe_r, pb_r, peb_r = pse.estimate(g1_r, g2_r)
    _, pe_shift, pb_shift, peb_shift = pse.estimate(g1_shift, g2_shift)

    # Check that they are identical.
    np.testing.assert_allclose(
        pe_shift, pe, rtol=1e-10,
        err_msg="E power altered by NN periodic interpolation.")
    np.testing.assert_allclose(
        pb_shift, pb, rtol=1e-10,
        err_msg="B power altered by NN periodic interpolation.")
    np.testing.assert_allclose(
        peb_shift, peb, rtol=1e-10,
        err_msg="EB power altered by NN periodic interpolation.")

    ### Check reduced shear ###
    g1_r_shift, g2_r_shift = ps.getShear(pos=(x.flatten(),y.flatten()), units=galsim.degrees,
                                         periodic=True)
    g1_r_shift = g1_r_shift.reshape((ngrid,ngrid))
    g2_r_shift = g2_r_shift.reshape((ngrid,ngrid))
    _, pe_r_shift, pb_r_shift, peb_r_shift = pse.estimate(g1_r_shift, g2_r_shift)
    np.testing.assert_allclose(
        pe_r_shift, pe_r, rtol=1e-10,
        err_msg="E power altered by NN periodic interpolation.")
    np.testing.assert_allclose(
        pb_r_shift, pb_r, rtol=1e-10,
        err_msg="B power altered by NN periodic interpolation.")
    np.testing.assert_allclose(
        peb_r_shift, peb_r, rtol=1e-10,
        err_msg="EB power altered by NN periodic interpolation.")

    ### Check getConvergence ###
    kappa_shift = ps.getConvergence(pos=(x.flatten(),y.flatten()),
                                    units=galsim.degrees, periodic=True)
    # We don't have a power spectrum measure, so let's just check the mean and variance.
    np.testing.assert_almost_equal(np.mean(kappa_shift), np.mean(kappa), decimal=8,
                                   err_msg='Mean convergence altered by periodic interpolation')
    np.testing.assert_almost_equal(np.var(kappa_shift), np.var(kappa), decimal=8,
                                   err_msg='Convergence variance altered by periodic interpolation')

    ### Check getMagnification ###
    mu_shift = ps.getMagnification(pos=(x.flatten(),y.flatten()),
                                   units=galsim.degrees, periodic=True)
    # We don't have a power spectrum measure, so let's just check the mean and variance.
    np.testing.assert_almost_equal(np.mean(mu_shift), np.mean(mu), decimal=8,
                                   err_msg='Mean magnification altered by periodic interpolation')
    np.testing.assert_almost_equal(np.var(mu_shift), np.var(mu), decimal=8,
                                   err_msg='Magnification variance altered by periodic interpolation')

    ### Now, check getLensing ###
    g1_r_shift, g2_r_shift, mu_shift = ps.getLensing(pos=(x.flatten(),y.flatten()),
                                                     units=galsim.degrees,
                                                     periodic=True)
    g1_r_shift = g1_r_shift.reshape((ngrid,ngrid))
    g2_r_shift = g2_r_shift.reshape((ngrid,ngrid))
    _, pe_r_shift, pb_r_shift, peb_r_shift = pse.estimate(g1_r_shift, g2_r_shift)
    np.testing.assert_allclose(
        pe_r_shift, pe_r, rtol=1e-10,
        err_msg="E power altered by NN periodic interpolation.")
    np.testing.assert_allclose(
        pb_r_shift, pb_r, rtol=1e-10,
        err_msg="B power altered by NN periodic interpolation.")
    np.testing.assert_allclose(
        peb_r_shift, peb_r, rtol=1e-10,
        err_msg="EB power altered by NN periodic interpolation.")
    # Should also check convergences/magnifications.  We don't have a power spectrum measure, so
    # let's just check the mean and variance.
    np.testing.assert_almost_equal(np.mean(mu_shift), np.mean(mu), decimal=8,
                                   err_msg='Mean magnification altered by periodic interpolation')
    np.testing.assert_almost_equal(np.var(mu_shift), np.var(mu), decimal=8,
                                   err_msg='Magnification variance altered by periodic interpolation')

    # If image is too small, can't use periodic boundaries.
    ps.buildGrid(ngrid=5, grid_spacing=0.1, units=galsim.degrees,
                 rng=galsim.UniformDeviate(314159), interpolant='lanczos7',
                 kmin_factor=3., kmax_factor=1., get_convergence=True)
    with assert_raises(galsim.GalSimError):
        ps.getShear(pos=(x.flatten(),y.flatten()), units=galsim.degrees,
                    reduced=False, periodic=True)

@timer
def test_bandlimit():
    """Test that the band-limiting of the power spectrum is working properly.
    """
    # If we do not impose a band limit on the power spectrum, then it's going to lead to aliasing in
    # both the E and B modes, which gives spurious power within kmin<k<kmax.   In practice this is
    # typically a 5-10% effect.  We are just going to test that the shear variance is suitably
    # elevated rather than testing the entire power spectrum.

    # Start with a cosmological power spectrum that is not band-limited.
    ps_tab = galsim.LookupTable.from_file('../examples/data/cosmo-fid.zmed1.00.out')
    ps = galsim.PowerSpectrum(ps_tab, units=galsim.radians)
    check_pickle(ps)

    # Generate shears without and with band-limiting
    g1, g2 = ps.buildGrid(ngrid=100, grid_spacing=0.1, units='degrees',
                          rng=galsim.UniformDeviate(123), bandlimit=None)
    g1b, g2b = ps.buildGrid(ngrid=100, grid_spacing=0.1, units=galsim.degrees,
                            rng=galsim.UniformDeviate(123), bandlimit='hard')

    # Check the shear variances for a >5% excess when there's no band limit.
    var = np.var(g1)+np.var(g2)
    varb = np.var(g1b)+np.var(g2b)
    assert var>1.05*varb,"Comparison of shear variances without/with band-limiting is not as expected"


@timer
def test_psr():
    """Test PowerSpectrumRealizer"""
    # Most of the tests of this class are implicit in its use by PowerSpectrum.
    # But since it is technically documented, we should make sure things like the repr and ==
    # work correctly.

    pe = galsim.LookupTable.from_file('../examples/data/cosmo-fid.zmed1.00.out')
    pb = galsim.LookupTable.from_file('../examples/data/cosmo-fid.zmed1.00_smoothed.out')
    psr = galsim.lensing_ps.PowerSpectrumRealizer(100, 0.005, pe, pb)
    check_pickle(psr)

    # Check ne
    assert psr == galsim.lensing_ps.PowerSpectrumRealizer(ngrid=100, pixel_size=0.005,
                                                          p_E=pe, p_B=pb)
    diff_psr_list = [psr,
                     galsim.lensing_ps.PowerSpectrumRealizer(50, 0.005, pe, pb),
                     galsim.lensing_ps.PowerSpectrumRealizer(100, 0.003, pe, pb),
                     galsim.lensing_ps.PowerSpectrumRealizer(100, 0.005, pe, None),
                     galsim.lensing_ps.PowerSpectrumRealizer(100, 0.005, None, pb),
                     galsim.lensing_ps.PowerSpectrumRealizer(100, 0.005, pb, pe)]
    check_all_diff(diff_psr_list)

    with assert_raises(TypeError):
        psr(gd=galsim.BaseDeviate(1234))


@timer
def test_normalization():
    """Test the normalization of the input power spectrum"""

    # Repeat some of this from test_shear_variance above.
    # We define it as P(k) = exp(-s^2 k^2 / 2).
    grid_size = 50. # degrees
    ngrid = 500
    kmin = 2.*np.pi/grid_size/3600.
    kmax = np.pi/(grid_size/ngrid)/3600.
    # Now choose s such that s*kmax=2.5, i.e., very little power at kmax.
    s = 2.5/kmax
    ps = galsim.PowerSpectrum(lambda k : np.exp(-0.5*((s*k)**2)))
    rng = galsim.BaseDeviate(12345)
    g1, g2, k = ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                             units=galsim.degrees, get_convergence=True)
    # The prediction for the variance is:
    # Var(g1) + Var(g2) = [1/(2 pi s^2)] * ( (Erf(s*kmax/sqrt(2)))^2 - (Erf(s*kmin/sqrt(2)))^2 )
    try:
        erfmax = math.erf(s*kmax/math.sqrt(2.))
        erfmin = math.erf(s*kmin/math.sqrt(2.))
        print('erfmax = ',erfmax)
        print('erfmin = ',erfmin)
    except: # For python2.6, which doesn't have math.erf.
        erfmax = 0.9875806693484477
        erfmin = 0.007978712629263206
    var1 = np.var(g1)
    var2 = np.var(g2)
    vark = np.var(k)
    print('varg = ',var1+var2)
    print('vark = ',vark)
    pred_var = (erfmax**2 - erfmin**2) / (2.*np.pi*(s**2))
    print('predicted variance = ',pred_var)
    print('actual variance = ',var1+var2)
    print('fractional diff = ',((var1+var2)/pred_var-1))
    np.testing.assert_allclose(var1+var2, pred_var, rtol=0.03,
                               err_msg="Incorrect shear variance from Gaussian power spectrum")
    np.testing.assert_allclose(vark, pred_var, rtol=0.03,
                               err_msg="Incorrect kappa variance from Gaussian power spectrum")

    # Renormalize to a given desired variance
    target_var = 0.04
    g1, g2, k = ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                             units=galsim.degrees, get_convergence=True, variance=target_var)
    var1 = np.var(g1)
    var2 = np.var(g2)
    vark = np.var(k)
    print('varg1 = ',var1)
    print('varg2 = ',var2)
    print('vark = ',vark)
    np.testing.assert_allclose(var1+var2, target_var, rtol=1.e-3,
                               err_msg="Incorrect shear variance using renormalized variance")
    np.testing.assert_allclose(vark, target_var, rtol=1.e-3,
                               err_msg="Incorrect kappa variance using renormalized variance")

    # Now do one that (AFAIK) doesn't have an analytic integral for the variance.
    # This is the kind of power spectrum that one expects for PSF shapes due to the atmosphere.
    L0 = 2.9 # arcmin.  Heymans et al, 2012 found L0 ~ 2.6 - 3.2 arcmin
    Pk = lambda k : 1.e-5 * (k**2 + 1/L0**2)**(-11/6)
    ps = galsim.PowerSpectrum(Pk, Pk, units=galsim.arcmin)

    grid_spacing = 30 # arcsec
    ngrid = 1000
    g1, g2, k = ps.buildGrid(ngrid=ngrid, grid_spacing=grid_spacing, units=galsim.arcsec,
                             rng=rng, get_convergence=True)

    var1 = np.var(g1)
    var2 = np.var(g2)
    vark = np.var(k)
    print('varg1 = ',var1)
    print('varg2 = ',var2)
    print('vark = ',vark)

    # Predicted variance is the integral of P(k) from kmin to kmax
    # As discussed in ../devel/modules/lensing_engine.pdf, this is not super accurate for
    # gridded power spectra, but in our case, with small grid spacing, it's not too bad.
    kmin = 2.*np.pi/(grid_spacing*ngrid*3600.)  # arcsec^-1
    kmax = np.pi/(grid_spacing/3600.)
    print('Pk(kmin) = ',Pk(kmin))
    print('Pk(kmax) = ',Pk(kmax))

    pred_var = galsim.integ.int1d(lambda k: 2*math.pi*k*Pk(k), min=kmin, max=kmax) / (2.*math.pi)**2
    print('pred_var = ',pred_var)
    print('ratio = ',pred_var / vark)
    np.testing.assert_allclose(var1+var2, 2*pred_var, rtol=0.01,
                               err_msg="Incorrect shear variance from atmospheric power spectrum")
    np.testing.assert_allclose(vark, pred_var, rtol=0.01,
                               err_msg="Incorrect kappa variance from atmospheric power spectrum")

    # Renormalize to a given desired variance
    target_var = 0.04
    g1, g2, k = ps.buildGrid(grid_spacing=grid_size/ngrid, ngrid=ngrid, rng=rng,
                             units=galsim.degrees, get_convergence=True, variance=target_var)
    var1 = np.var(g1)
    var2 = np.var(g2)
    vark = np.var(k)
    print('varg1 = ',var1)
    print('varg2 = ',var2)
    print('vark = ',vark)
    np.testing.assert_allclose(var1+var2, target_var, rtol=1.e-3,
                               err_msg="Incorrect shear variance using renormalized variance")
    np.testing.assert_allclose(vark, target_var/2., rtol=1.e-3,
                               err_msg="Incorrect kappa variance using renormalized variance")

@timer
def test_constant():
    """Test P(k) = constant"""
    # We used to require that P(k) return an array, but that's not required anymore.
    ps_a = galsim.PowerSpectrum(lambda k: 4, units=galsim.arcsec)
    ps_b = galsim.PowerSpectrum(lambda k: 4.*np.ones_like(k), units=galsim.arcsec)
    rng_a = galsim.BaseDeviate(1234)
    rng_b = galsim.BaseDeviate(1234)
    g1_a, g2_a, k_a = ps_a.buildGrid(ngrid=100, grid_spacing=0.1, units=galsim.degrees,
                                     rng=rng_a, get_convergence=True)
    g1_b, g2_b, k_b = ps_b.buildGrid(ngrid=100, grid_spacing=0.1, units=galsim.degrees,
                                     rng=rng_b, get_convergence=True)
    np.testing.assert_allclose(g1_a, g1_b, rtol=1.e-10)
    np.testing.assert_allclose(g2_a, g2_b, rtol=1.e-10)
    np.testing.assert_allclose(k_a, k_b, rtol=1.e-10)

    # Repeat with bandlimit = None, since that's the only one that could actually be a problem.
    g1_a, g2_a, k_a = ps_a.buildGrid(ngrid=100, grid_spacing=0.1, units=galsim.degrees,
                                     rng=rng_a, get_convergence=True, bandlimit=None)
    g1_b, g2_b, k_b = ps_b.buildGrid(ngrid=100, grid_spacing=0.1, units=galsim.degrees,
                                     rng=rng_b, get_convergence=True, bandlimit=None)
    np.testing.assert_allclose(g1_a, g1_b, rtol=1.e-10)
    np.testing.assert_allclose(g2_a, g2_b, rtol=1.e-10)
    np.testing.assert_allclose(k_a, k_b, rtol=1.e-10)

    # And might as well hit soft as well for good measure
    g1_a, g2_a, k_a = ps_a.buildGrid(ngrid=100, grid_spacing=0.1, units=galsim.degrees,
                                     rng=rng_a, get_convergence=True, bandlimit='soft')
    g1_b, g2_b, k_b = ps_b.buildGrid(ngrid=100, grid_spacing=0.1, units=galsim.degrees,
                                     rng=rng_b, get_convergence=True, bandlimit='soft')
    np.testing.assert_allclose(g1_a, g1_b, rtol=1.e-10)
    np.testing.assert_allclose(g2_a, g2_b, rtol=1.e-10)
    np.testing.assert_allclose(k_a, k_b, rtol=1.e-10)


if __name__ == "__main__":
    runtests(__file__)
