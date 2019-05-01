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


@timer
def test_Zernike_orthonormality():
    r""" Zernike optical screens *should* be normalized such that
    \int_{unit disc} Z(n1, m1) Z(n2, m2) dA = \pi in unit disc coordinates, or alternatively
    = aperture area if radial coordinate is not normalized (i.e., diam != 2).
    """
    jmax = 30  # Going up to 30 filled Zernikes takes about ~1 sec on my laptop
    diam = 4.0
    R_outer = diam/2

    x = np.linspace(-R_outer, R_outer, 256)
    dx = x[1]-x[0]
    x, y = np.meshgrid(x, x)
    w = np.hypot(x, y) <= R_outer
    x = x[w].ravel()
    y = y[w].ravel()
    area = np.pi*R_outer**2
    for j1 in range(1, jmax+1):
        Z1 = galsim.zernike.Zernike([0]*(j1+1)+[1], R_outer=R_outer)
        val1 = Z1.evalCartesian(x, y)
        for j2 in range(j1, jmax+1):
            Z2 = galsim.zernike.Zernike([0]*(j2+1)+[1], R_outer=R_outer)
            val2 = Z2.evalCartesian(x, y)
            integral = np.dot(val1, val2) * dx**2
            if j1 == j2:
                # Only passes at ~1% level because of pixelization.
                np.testing.assert_allclose(
                        integral, area, rtol=1e-2,
                        err_msg="Orthonormality failed for (j1,j2) = ({0},{1})".format(j1, j2))
            else:
                # Only passes at ~1% level because of pixelization.
                np.testing.assert_allclose(
                        integral, 0.0, atol=area*1e-2,
                        err_msg="Orthonormality failed for (j1,j2) = ({0},{1})".format(j1, j2))

    do_pickle(Z1)
    do_pickle(Z1, lambda z: tuple(z.evalCartesian(x, y)))

    # Repeat for Annular Zernikes
    jmax = 22  # Going up to 22 annular Zernikes takes about ~1 sec on my laptop
    R_inner = 0.6
    x = np.linspace(-R_outer, R_outer, 256)
    dx = x[1]-x[0]
    x, y = np.meshgrid(x, x)
    r = np.hypot(x, y)
    w = np.logical_and(R_inner <= r, r <= R_outer)
    x = x[w].ravel()
    y = y[w].ravel()
    area = np.pi*(R_outer**2 - R_inner**2)
    for j1 in range(1, jmax+1):
        Z1 = galsim.zernike.Zernike([0]*(j1+1)+[1], R_outer=R_outer, R_inner=R_inner)
        val1 = Z1.evalCartesian(x, y)
        for j2 in range(j1, jmax+1):
            Z2 = galsim.zernike.Zernike([0]*(j2+1)+[1], R_outer=R_outer, R_inner=R_inner)
            val2 = Z2.evalCartesian(x, y)
            integral = np.dot(val1, val2) * dx**2
            if j1 == j2:
                # Only passes at ~1% level because of pixelization.
                np.testing.assert_allclose(
                        integral, area, rtol=1e-2,
                        err_msg="Orthonormality failed for (j1,j2) = ({0},{1})".format(j1, j2))
            else:
                # Only passes at ~1% level because of pixelization.
                np.testing.assert_allclose(
                        integral, 0.0, atol=area*1e-2,
                        err_msg="Orthonormality failed for (j1,j2) = ({0},{1})".format(j1, j2))
    do_pickle(Z1)
    do_pickle(Z1, lambda z: tuple(z.evalCartesian(x, y)))

    with assert_raises(ValueError):
        Z1 = galsim.zernike.Zernike([0]*4 + [0.1]*7, R_outer=R_inner, R_inner=R_outer)
        val1 = Z1.evalCartesian(x, y)


@timer
def test_annular_Zernike_limit():
    """Check that annular Zernike matches circular Zernike in the limit of 0.0 obscuration.
    """
    jmax = 20
    bd = galsim.BaseDeviate(1029384756)
    u = galsim.UniformDeviate(bd)

    diam = 4.0

    for i in range(4):  # Do a few random tests.  Takes about 1 sec.
        aberrations = [0]+[u() for i in range(jmax)]
        psf1 = galsim.OpticalPSF(diam=diam, lam=500, obscuration=1e-5,
                                 aberrations=aberrations, annular_zernike=True)
        psf2 = galsim.OpticalPSF(diam=diam, lam=500, obscuration=1e-5,
                                 aberrations=aberrations)
        im1 = psf1.drawImage()
        im2 = psf2.drawImage(image=im1.copy())
        # We want the images to be close, since the obscuration is near 0, but not identical.
        # That way we know that the `annular_zernike` keyword is doing something.
        assert im1 != im2, "annular Zernike identical to circular Zernike"
        np.testing.assert_allclose(
                im1.array, im2.array, atol=1e-10,
                err_msg="annular Zernike with 1e-5 obscuration not close to circular Zernike")

    do_pickle(psf1._aper)
    do_pickle(psf1)
    do_pickle(psf1, lambda x: x.drawImage())
    do_pickle(psf2)
    do_pickle(psf2, lambda x: x.drawImage())


@timer
def test_noll():
    # This function stolen from https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
    # It used to be in zernike.py, but now we use a faster lookup-table implementation.
    # This reference version is still useful as a test.
    def noll_to_zern(j):
        if (j == 0):
            raise ValueError("Noll indices start at 1. 0 is invalid.")
        n = 0
        j1 = j-1
        while (j1 > n):
            n += 1
            j1 -= n
        m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1) % 2)) / 2.0))
        return (n, m)

    # Test that the version of noll_to_zern in zernike.py is accurate.
    for j in range(1,30):
        true_n,true_m = noll_to_zern(j)
        n,m = galsim.zernike.noll_to_zern(j)
        #print('j=%d, noll = %d,%d, true_noll = %d,%d'%(j,n,m,true_n,true_m))
        assert n == true_n
        assert m == true_m
        # These didn't turn out to be all that useful for fast conversions, but they're cute.
        assert n == int(np.sqrt(8*j-7)-1)//2
        mm = -m if (n//2)%2 == 0 else m
        assert j == n*(n+1)/2 + (abs(2*mm+1)+1)//2

    # Again, the reference version of this function used to be in zernike.py
    def zern_rho_coefs(n, m):
        """Compute coefficients of radial part of Zernike (n, m).
        """
        from galsim.utilities import nCr
        kmax = (n-abs(m)) // 2
        A = np.zeros(n+1)
        for k in range(kmax+1):
            A[n-2*k] = (-1)**k * nCr(n-k, k) * nCr(n-2*k, kmax-k)
        return A

    for j in range(1,30):
        n,m = galsim.zernike.noll_to_zern(j)
        true_coefs = zern_rho_coefs(n,m)
        coefs = galsim.zernike._zern_rho_coefs(n,m)
        #print('j=%d, coefs = %s'%(j,coefs))
        np.testing.assert_array_equal(coefs,true_coefs)


@timer
def test_Zernike_rotate():
    """Test that rotating Zernike coefficients to another coord sys works as expected"""
    #First check that invalid Zernike rotation matrix sizes are trapped
    with assert_raises(ValueError):
        # Can't do size=2, since Z2 mixes into Z3
        galsim.zernike.zernikeRotMatrix(2, 0.1)
        # Can't do size=5, since Z5 mixes into Z6
        galsim.zernike.zernikeRotMatrix(5, 0.2)

    u = galsim.UniformDeviate(12020569031)
    #Now let's test some actual rotations.
    for jmax in [1, 3, 10, 11, 13, 21, 22, 34]:
        # Pick some arbitrary eps and diams
        eps = (jmax % 5)/10.0
        diam = ((jmax % 10) + 1.0)
        # Test points
        rhos = np.linspace(0, diam/2, 4)
        thetas = np.linspace(0, np.pi, 4)

        R_outer = diam/2.0
        R_inner = R_outer*eps

        coefs = [u() for _ in range(jmax+1)]
        Z = galsim.zernike.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
        do_pickle(Z)

        for theta in [0.0, 0.1, 1.0, np.pi, 4.0]:
            R = galsim.zernike.zernikeRotMatrix(jmax, theta)
            rotCoefs = np.dot(R, coefs)
            Zrot = galsim.zernike.Zernike(rotCoefs, R_outer=R_outer, R_inner=R_inner)
            np.testing.assert_allclose(
                Z.evalPolar(rhos, thetas),
                Zrot.evalPolar(rhos, thetas+theta),
                atol=1e-13, rtol=0
            )

            Zrot2 = Z.rotate(theta)
            np.testing.assert_allclose(
                Z.evalPolar(rhos, thetas),
                Zrot2.evalPolar(rhos, thetas+theta),
                atol=1e-13, rtol=0
            )


@timer
def test_ne():
    objs = [
        galsim.zernike.Zernike([0, 1, 2]),
        galsim.zernike.Zernike([0, 1, 2, 3]),
        galsim.zernike.Zernike([0, 1, 2, 3], R_outer=0.2),
        galsim.zernike.Zernike([0, 1, 2, 3], R_outer=0.2, R_inner=0.1),
    ]
    all_obj_diff(objs)


@timer
def test_Zernike_basis():
    """Test the zernikeBasis function"""
    diam = 2.4
    jmax = 30
    R_outer = diam/2
    R_inner = R_outer*0.2

    u = galsim.UniformDeviate(4669201609)
    for i in range(10):
        # Test at some random points
        x = np.empty((10000,), dtype=np.float)
        y = np.empty((10000,), dtype=np.float)
        u.generate(x)
        u.generate(y)

        # zBases will generate all basis vectors at once
        zBases = galsim.zernike.zernikeBasis(jmax, x, y, R_outer=R_outer, R_inner=R_inner)

        # Compare to basis vectors generated one at a time
        for j in range(1, jmax):
            Z = galsim.zernike.Zernike([0]*(j-1)+[1], R_outer=R_outer, R_inner=R_inner)
            zBasis = Z.evalCartesian(x, y)
            np.testing.assert_allclose(
                    zBases[j-1],
                    zBasis,
                    atol=1e-12, rtol=0)


@timer
def test_fit():
    """Test fitting values to a Zernike series, using the ZernikeBasis function"""
    u = galsim.UniformDeviate(161803)
    for i in range(10):
        x = np.empty((1000,), dtype=np.float)
        y = np.empty((1000,), dtype=np.float)
        u.generate(x)
        u.generate(y)
        x -= 0.5
        y -= 0.5
        R_outer = (i%5/5.0)+1
        R_inner = ((i%3/6.0)+0.1)*(R_outer)
        x *= R_outer
        y *= R_outer

        # Should be able to fit quintic polynomial by including Zernikes up to Z_21
        cartesian_coefs = [[u()-0.5, u()-0.5, u()-0.5, u()-0.5, u()-0.5],
                           [u()-0.5, u()-0.5, u()-0.5, u()-0.5,       0],
                           [u()-0.5, u()-0.5, u()-0.5,       0,       0],
                           [u()-0.5, u()-0.5,       0,       0,       0],
                           [u()-0.5,       0,       0,       0,       0]]
        z = galsim.utilities.horner2d(x, y, cartesian_coefs)
        z2 = galsim.utilities.horner2d(x, y, cartesian_coefs, triangle=True)
        np.testing.assert_equal(z,z2)

        basis = galsim.zernike.zernikeBasis(21, x, y, R_outer=R_outer, R_inner=R_inner)
        coefs, _, _, _ = np.linalg.lstsq(basis.T, z, rcond=-1.)
        resids = (galsim.zernike.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
                  .evalCartesian(x, y)
                  - z)
        resids2 = np.dot(basis.T, coefs).T - z
        assert resids.shape == x.shape
        assert resids2.shape == x.shape

        np.testing.assert_allclose(resids, 0, atol=1e-14)
        np.testing.assert_allclose(resids2, 0, atol=1e-14)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
        # scat1 = axes[0].scatter(x, y, c=z)
        # plt.colorbar(scat1, ax=axes[0])
        # scat2 = axes[1].scatter(x, y, c=resids)
        # plt.colorbar(scat2, ax=axes[1])
        # plt.show()
        # print(np.mean(resids), np.std(resids))

    # Should also work, and make congruent output, if the shapes of x and y are multi-dimensional
    for i in range(10):
        x = np.empty((1000,), dtype=np.float)
        y = np.empty((1000,), dtype=np.float)
        u.generate(x)
        u.generate(y)
        x -= 0.5
        y -= 0.5
        R_outer = (i%5/5.0)+1
        R_inner = ((i%3/6.0)+0.1)*(R_outer)
        x *= R_outer
        y *= R_outer
        x = x.reshape(25, 40)
        y = y.reshape(25, 40)

        # Should be able to fit quintic polynomial by including Zernikes up to Z_21
        cartesian_coefs = [[u()-0.5, u()-0.5, u()-0.5, u()-0.5, u()-0.5],
                           [u()-0.5, u()-0.5, u()-0.5, u()-0.5,       0],
                           [u()-0.5, u()-0.5, u()-0.5,       0,       0],
                           [u()-0.5, u()-0.5,       0,       0,       0],
                           [u()-0.5,       0,       0,       0,       0]]
        z = galsim.utilities.horner2d(x, y, cartesian_coefs)
        assert z.shape == (25, 40)
        z2 = galsim.utilities.horner2d(x, y, cartesian_coefs, triangle=True)
        np.testing.assert_equal(z,z2)

        basis = galsim.zernike.zernikeBasis(21, x, y, R_outer=R_outer, R_inner=R_inner)
        assert basis.shape == (22, 25, 40)
        # lstsq doesn't handle the extra dimension though...
        coefs, _, _, _ = np.linalg.lstsq(basis.reshape(21+1, 1000).T, z.ravel(), rcond=-1.)
        resids = (galsim.zernike.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
                  .evalCartesian(x, y)
                  - z)
        resids2 = np.dot(basis.T, coefs).T - z
        assert resids.shape == resids2.shape == x.shape

        np.testing.assert_allclose(resids, 0, atol=1e-14)
        np.testing.assert_allclose(resids2, 0, atol=1e-14)


@timer
def test_gradient():
    # Start with a few that just quote the literature, e.g., Stephenson (2014).

    Z11 = galsim.zernike.Zernike([0]*11+[1])

    x = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, x)

    def Z11_grad(x, y):
        # Z11 = sqrt(5) (6(x^2+y^2)^2 - 6(x^2+y^2)+1)
        r2 = x**2 + y**2
        gradx = 12*np.sqrt(5)*x*(2*r2-1)
        grady = 12*np.sqrt(5)*y*(2*r2-1)
        return gradx, grady

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(ncols=3, figsize=(12, 3))
    # scat0 = axes[0].scatter(x, y, c=Z11.evalCartesianGrad(x, y)[0])
    # fig.colorbar(scat0, ax=axes[0])
    # scat1 = axes[1].scatter(x, y, c=Z11_grad(x, y)[0])
    # fig.colorbar(scat1, ax=axes[1])
    # scat2 = axes[2].scatter(x, y, c=Z11.evalCartesianGrad(x, y)[0] - Z11_grad(x, y)[0])
    # fig.colorbar(scat2, ax=axes[2])
    # plt.show()

    np.testing.assert_allclose(Z11.evalCartesianGrad(x, y), Z11_grad(x, y), rtol=0, atol=1e-12)

    Z28 = galsim.zernike.Zernike([0]*28+[1])

    def Z28_grad(x, y):
        # Z28 = sqrt(14) (x^6 - 15 x^4 y^2 + 15 x^2 y^4 - y^6)
        gradx = 6*np.sqrt(14)*x*(x**4 - 10*x**2*y**2 + 5*y**4)
        grady = -6*np.sqrt(14)*y*(5*x**4 - 10*x**2*y**2 + y**4)
        return gradx, grady

    np.testing.assert_allclose(Z28.evalCartesianGrad(x, y), Z28_grad(x, y), rtol=0, atol=1e-12)

    # Now try some finite differences on a broader set of input

    def finite_difference_gradient(Z, x, y):
        dh = 1e-5
        return ((Z.evalCartesian(x+dh, y)-Z.evalCartesian(x-dh, y))/(2*dh),
                (Z.evalCartesian(x, y+dh)-Z.evalCartesian(x, y-dh))/(2*dh))

    u = galsim.UniformDeviate(1234)

    # Test finite difference against analytic result for 25 different Zernikes with random number of
    # random coefficients and random inner/outer radii.
    for j in range(25):
        nj = 1+int(u()*55)
        R_inner = 0.2+0.6*u()
        R_outer = R_inner + 0.2+0.6*u()
        Z = galsim.zernike.Zernike([0]+[u() for _ in range(nj)], R_inner=R_inner, R_outer=R_outer)

        np.testing.assert_allclose(
                finite_difference_gradient(Z, x, y),
                Z.evalCartesianGrad(x, y),
                rtol=1e-5, atol=1e-5)

    # Make sure the gradient of the zero-Zernike works
    Z = galsim.zernike.Zernike([0])
    assert Z == Z.gradX == Z.gradX.gradX == Z.gradY == Z.gradY.gradY

if __name__ == "__main__":
    test_Zernike_orthonormality()
    test_annular_Zernike_limit()
    test_noll()
    test_Zernike_rotate()
    test_ne()
    test_Zernike_basis()
    test_fit()
    test_gradient()
