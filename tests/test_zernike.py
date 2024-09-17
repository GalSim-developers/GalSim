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

import galsim
from galsim.zernike import Zernike, DoubleZernike
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
        Z1 = Zernike([0]*(j1+1)+[1], R_outer=R_outer)
        val1 = Z1.evalCartesian(x, y)
        for j2 in range(j1, jmax+1):
            Z2 = Zernike([0]*(j2+1)+[1], R_outer=R_outer)
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

    check_pickle(Z1)
    check_pickle(Z1, lambda z: tuple(z.evalCartesian(x, y)))

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
        Z1 = Zernike([0]*(j1+1)+[1], R_outer=R_outer, R_inner=R_inner)
        val1 = Z1.evalCartesian(x, y)
        for j2 in range(j1, jmax+1):
            Z2 = Zernike([0]*(j2+1)+[1], R_outer=R_outer, R_inner=R_inner)
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
    check_pickle(Z1)
    check_pickle(Z1, lambda z: tuple(z.evalCartesian(x, y)))

    with assert_raises(ValueError):
        Z1 = Zernike([0]*4 + [0.1]*7, R_outer=R_inner, R_inner=R_outer)
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

    check_pickle(psf1._aper)
    check_pickle(psf1)
    check_pickle(psf1, lambda x: x.drawImage())
    check_pickle(psf2)
    check_pickle(psf2, lambda x: x.drawImage())


@timer
def test_noll():
    """Test that Noll indexing scheme between j <-> (n,m) works as expected.
    """
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
        Z = Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
        check_pickle(Z)

        for theta in [0.0, 0.1, 1.0, np.pi, 4.0]:
            R = galsim.zernike.zernikeRotMatrix(jmax, theta)
            rotCoefs = np.dot(R, coefs)
            Zrot = Zernike(rotCoefs, R_outer=R_outer, R_inner=R_inner)
            print('j,theta: ',jmax,theta)
            print('Z: ',Z.evalPolar(rhos, thetas))
            print('Zrot: ',Zrot.evalPolar(rhos, thetas+theta))
            print('max diff= ',np.max(np.abs(Zrot.evalPolar(rhos, thetas+theta)-Z.evalPolar(rhos, thetas))))
            np.testing.assert_allclose(
                Z.evalPolar(rhos, thetas),
                Zrot.evalPolar(rhos, thetas+theta),
                atol=1e-12, rtol=0
            )

            Zrot2 = Z.rotate(theta)
            print('Zrot2: ',Zrot2.evalPolar(rhos, thetas+theta))
            print('max diff= ',np.max(np.abs(Zrot2.evalPolar(rhos, thetas+theta)-Z.evalPolar(rhos, thetas))))
            np.testing.assert_allclose(
                Z.evalPolar(rhos, thetas),
                Zrot2.evalPolar(rhos, thetas+theta),
                atol=1e-12, rtol=0
            )


@timer
def test_zernike_eval():
    for coef in [
        np.ones(4),
        np.ones(4, dtype=float),
        np.ones(4, dtype=np.float32)
    ]:
        Z = Zernike(coef)
        assert Z.coef.dtype == np.float64
        assert Z(0.0, 0.0) == 1.0
        assert Z(0, 0) == 1.0

    for coefs in [
        np.ones((4, 4)),
        np.ones((4, 4), dtype=float),
        np.ones((4, 4), dtype=np.float32)
    ]:
        dz = DoubleZernike(coefs)
        assert dz.coef.dtype == np.float64
        assert dz(0.0, 0.0) == dz(0, 0)

        # Make sure we cast to float in _from_uvxy
        uvxy = dz._coef_array_uvxy
        dz2 = DoubleZernike._from_uvxy(uvxy.astype(int))
        np.testing.assert_array_equal(dz2._coef_array_uvxy, dz._coef_array_uvxy)


@timer
def test_ne():
    objs = [
        Zernike([0, 1, 2]),
        Zernike([0, 1, 2, 3]),
        Zernike([0, 1, 2, 3], R_outer=0.2),
        Zernike([0, 1, 2, 3], R_outer=0.2, R_inner=0.1),
        DoubleZernike(np.eye(3)),
        DoubleZernike(np.ones((4, 4))),
        DoubleZernike(np.ones((4, 4)), xy_outer=1.1),
        DoubleZernike(np.ones((4, 4)), xy_outer=1.1, xy_inner=0.9),
        DoubleZernike(np.ones((4, 4)), xy_outer=1.1, xy_inner=0.9, uv_outer=1.1),
        DoubleZernike(np.ones((4, 4)), xy_outer=1.1, xy_inner=0.9, uv_outer=1.1, uv_inner=0.9)
    ]
    check_all_diff(objs)


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
        x = np.empty((10000,), dtype=float)
        y = np.empty((10000,), dtype=float)
        u.generate(x)
        u.generate(y)

        # zBases will generate all basis vectors at once
        zBases = galsim.zernike.zernikeBasis(jmax, x, y, R_outer=R_outer, R_inner=R_inner)

        # Compare to basis vectors generated one at a time
        for j in range(1, jmax):
            Z = Zernike([0]*j+[1], R_outer=R_outer, R_inner=R_inner)
            zBasis = Z.evalCartesian(x, y)
            np.testing.assert_allclose(
                    zBases[j],
                    zBasis,
                    atol=1e-12, rtol=0)


@timer
def test_fit():
    """Test fitting values to a Zernike series, using the ZernikeBasis function"""
    u = galsim.UniformDeviate(161803)
    for i in range(10):
        x = np.empty((1000,), dtype=float)
        y = np.empty((1000,), dtype=float)
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
        resids = (Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
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
        x = np.empty((1000,), dtype=float)
        y = np.empty((1000,), dtype=float)
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
        resids = (Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
                  .evalCartesian(x, y)
                  - z)
        resids2 = np.dot(basis.T, coefs).T - z
        assert resids.shape == resids2.shape == x.shape

        np.testing.assert_allclose(resids, 0, atol=1e-14)
        np.testing.assert_allclose(resids2, 0, atol=1e-14)


@timer
def test_gradient():
    """Test that .gradX and .gradY properties work as expected.
    """
    # Start with a few that just quote the literature, e.g., Stephenson (2014).

    Z11 = Zernike([0]*11+[1])

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

    np.testing.assert_allclose(Z11.evalCartesianGrad(x, y), Z11_grad(x, y), rtol=1.e-12, atol=1e-12)

    Z28 = Zernike([0]*28+[1])

    def Z28_grad(x, y):
        # Z28 = sqrt(14) (x^6 - 15 x^4 y^2 + 15 x^2 y^4 - y^6)
        gradx = 6*np.sqrt(14)*x*(x**4 - 10*x**2*y**2 + 5*y**4)
        grady = -6*np.sqrt(14)*y*(5*x**4 - 10*x**2*y**2 + y**4)
        return gradx, grady

    np.testing.assert_allclose(Z28.evalCartesianGrad(x, y), Z28_grad(x, y), rtol=1.e-12, atol=1e-12)

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
        Z = Zernike([0]+[u() for _ in range(nj)], R_inner=R_inner, R_outer=R_outer)

        np.testing.assert_allclose(
                finite_difference_gradient(Z, x, y),
                Z.evalCartesianGrad(x, y),
                rtol=1e-5, atol=1e-5)

    # Make sure the gradient of the zero-Zernike works
    Z = Zernike([0])
    assert Z == Z.gradX == Z.gradX.gradX == Z.gradY == Z.gradY.gradY


@timer
def test_gradient_bases():
    """Test the zernikeGradBases function"""
    diam = 2.4
    jmax = 36
    R_outer = diam/2
    R_inner = R_outer*0.2

    u = galsim.UniformDeviate(1029384756)
    for i in range(10):
        # Test at some random points
        x = np.empty((10000,), dtype=float)
        y = np.empty((10000,), dtype=float)
        u.generate(x)
        u.generate(y)

        dxBasis, dyBasis = galsim.zernike.zernikeGradBases(
            jmax, x, y, R_outer=R_outer, R_inner=R_inner
        )

        # Compare to basis vectors generated one at a time
        for j in range(1, jmax+1):
            Z = Zernike([0]*j+[1], R_outer=R_outer, R_inner=R_inner)
            ZX = Z.gradX
            ZY = Z.gradY

            dx = ZX.evalCartesian(x, y)
            dy = ZY.evalCartesian(x, y)

            np.testing.assert_allclose(
                dx, dxBasis[j],
                atol=1e-11, rtol=1e-11
            )

            np.testing.assert_allclose(
                dy, dyBasis[j],
                atol=1e-11, rtol=1e-11
            )


@timer
def test_sum():
    """Test that __add__, __sub__, and __neg__ all work as expected.
    """
    u = galsim.UniformDeviate(5)
    x = np.empty(100, dtype=float)
    y = np.empty(100, dtype=float)
    u.generate(x)
    u.generate(y)

    for _ in range(100):
        n1 = int(u()*53)+3
        n2 = int(u()*53)+3
        R_outer = 1+0.1*u()
        R_inner = 0.1*u()
        if n1 > n2:
            n1, n2 = n2, n1
        a1 = np.empty(n1, dtype=float)
        a2 = np.empty(n2, dtype=float)
        u.generate(a1)
        u.generate(a2)
        z1 = Zernike(a1, R_outer=R_outer, R_inner=R_inner)
        z2 = Zernike(a2, R_outer=R_outer, R_inner=R_inner)

        c1 = u()
        c2 = u()

        coefSum = c2*np.array(z2.coef)
        coefSum[:len(z1.coef)] += c1*z1.coef
        coefDiff = c2*np.array(z2.coef)
        coefDiff[:len(z1.coef)] -= c1*z1.coef
        np.testing.assert_allclose(coefSum, (c1*z1 + c2*z2).coef)
        np.testing.assert_allclose(coefDiff, -(c1*z1 - c2*z2).coef)

        np.testing.assert_allclose(
            c1*z1(x, y) + c2*z2(x, y),
            (c1*z1 + c2*z2)(x, y)
        )
        np.testing.assert_allclose(
            c1*z1(x, y) - c2*z2(x, y),
            (c1*z1 - c2*z2)(x, y)
        )
        # Check that R_outer and R_inner are preserved
        np.testing.assert_allclose(
            (z1+z2).R_outer,
            R_outer
        )
        np.testing.assert_allclose(
            (z1+z2).R_inner,
            R_inner
        )

    with np.testing.assert_raises(TypeError):
        z1 + 3
    with np.testing.assert_raises(TypeError):
        z1 - 3
    with np.testing.assert_raises(ValueError):
        z1 + Zernike([0,1], R_outer=z1.R_outer*2)
    with np.testing.assert_raises(ValueError):
        z1 + Zernike([0,1], R_outer=z1.R_outer, R_inner=z1.R_inner*2)

    # Commutative with integer coefficients
    z1 = Zernike([0,1,2,3,4])
    z2 = Zernike([1,2,3,4,5,6])
    assert z1+z2 == z2+z1
    assert (z2-z1) == z2 + -z1 == -(z1-z2)


@timer
def test_product():
    """Test that __mul__ and __rmul__ work as expected.
    """
    u = galsim.UniformDeviate(57)
    x = np.empty(100, dtype=float)
    y = np.empty(100, dtype=float)
    u.generate(x)
    u.generate(y)

    for _ in range(100):
        n1 = int(u()*21)+3
        n2 = int(u()*21)+3
        R_outer = 1+0.1*u()
        R_inner = 0.1*u()
        a1 = np.empty(n1, dtype=float)
        a2 = np.empty(n2, dtype=float)
        u.generate(a1)
        u.generate(a2)
        z1 = Zernike(a1, R_outer=R_outer, R_inner=R_inner)
        z2 = Zernike(a2, R_outer=R_outer, R_inner=R_inner)

        np.testing.assert_allclose(
            z1(x, y) * z2(x, y),
            (z1 * z2)(x, y),
        )
        np.testing.assert_allclose(
            z1(x, y) * z2(x, y),
            (z2 * z1)(x, y),
        )
        # Check scalar multiplication
        np.testing.assert_allclose(
            (2*z1)(x, y),
            2*(z1(x, y)),
        )
        np.testing.assert_allclose(
            (z1*3.3)(x, y),
            3.3*(z1(x, y)),
        )
        # Check when .coef is missing
        del z1.coef
        np.testing.assert_allclose(
            (z1*3.5)(x, y),
            3.5*(z1(x, y)),
        )
        # Check that R_outer and R_inner are preserved
        np.testing.assert_allclose(
            (z1*z2).R_outer,
            R_outer
        )
        np.testing.assert_allclose(
            (z1*z2).R_inner,
            R_inner
        )
        # Check div
        np.testing.assert_allclose(
            (z1/5.6)(x, y),
            z1(x, y)/5.6,
        )

    with np.testing.assert_raises(TypeError):
        z1 * galsim.Gaussian(fwhm=1)
    with np.testing.assert_raises(ValueError):
        z1 * Zernike([0,1], R_outer=z1.R_outer*2)
    with np.testing.assert_raises(ValueError):
        z1 * Zernike([0,1], R_outer=z1.R_outer, R_inner=z1.R_inner*2)
    with np.testing.assert_raises(TypeError):
        z1 / z2

    # Commutative with integer coefficients
    z1 = Zernike([0,1,2,3,4,5])
    z2 = Zernike([1,2,3,4,5,6])
    assert z1*z2 == z2*z1


@timer
def test_laplacian():
    """Test .laplacian property.
    """
    u = galsim.UniformDeviate(577)
    x = np.empty(100, dtype=float)
    y = np.empty(100, dtype=float)
    u.generate(x)
    u.generate(y)

    for _ in range(200):
        n = int(u()*21)+3
        a = np.empty(n, dtype=float)
        u.generate(a)
        R_outer = 1+0.1*u()
        R_inner = 0.1*u()
        z = Zernike(a, R_outer=R_outer, R_inner=R_inner)

        np.testing.assert_allclose(
            z.laplacian(x, y),
            z.gradX.gradX(x, y) + z.gradY.gradY(x, y)
        )
        # Check that R_outer and R_inner are preserved
        np.testing.assert_allclose(
            z.laplacian.R_outer,
            R_outer
        )
        np.testing.assert_allclose(
            z.laplacian.R_inner,
            R_inner
        )

    # Do a couple by hand
    # Z4 = sqrt(3) (2x^2 + 2y^2 - 1)
    # implies laplacian = 4 sqrt(3) + 4 sqrt(3) = 8 sqrt(3)
    # which is 8 sqrt(3) Z1
    np.testing.assert_allclose(
        Zernike([0,0,0,0,1]).laplacian.coef,
        np.array([0,8*np.sqrt(3)])
    )

    # Z7 = sqrt(8) * (3 * (x^2 + y^2) - 2) * y
    # implies d^2/dx^2 = 6 sqrt(8) y
    #         d^2/dy^2 = 12 sqrt(8) y + 6 sqrt(8) y = 18 sqrt(8) y
    # implies laplacian = 24 sqrt(8) y
    # which is 12*sqrt(8) * Z3 since Z3 = 2 y
    np.testing.assert_allclose(
        Zernike([0,0,0,0,0,0,0,1]).laplacian.coef,
        np.array([0,0,0,12*np.sqrt(8)])
    )


@timer
def test_hessian():
    """Test .hessian property.
    """
    u = galsim.UniformDeviate(5772)
    x = np.empty(100, dtype=float)
    y = np.empty(100, dtype=float)
    u.generate(x)
    u.generate(y)

    for _ in range(200):
        n = int(u()*21)+3
        a = np.empty(n, dtype=float)
        u.generate(a)
        R_outer = 1+0.1*u()
        R_inner = 0.1*u()
        z = Zernike(a, R_outer=R_outer, R_inner=R_inner)

        np.testing.assert_allclose(
            z.hessian(x, y),
            z.gradX.gradX(x, y) * z.gradY.gradY(x, y) - z.gradX.gradY(x, y)**2
        )
        # Check that R_outer and R_inner are preserved
        np.testing.assert_allclose(
            z.hessian.R_outer,
            R_outer
        )
        np.testing.assert_allclose(
            z.hessian.R_inner,
            R_inner
        )

    # Do a couple by hand
    # Z4 = sqrt(3) (2x^2 + 2y^2 - 1)
    # implies hessian = 4 sqrt(3) * 4 sqrt(3) - 0 * 0 = 16*3 = 48
    # which is 48 Z1
    np.testing.assert_allclose(
        Zernike([0,0,0,0,1]).hessian.coef,
        np.array([0,48])
    )

    # Z7 = sqrt(8) * (3 * (x^2 + y^2) - 2) * y
    # implies d^2/dx^2 = 6 sqrt(8) y
    #         d^2/dy^2 = 12 sqrt(8) y + 6 sqrt(8) y = 18 sqrt(8) y
    #         d^2/dxdy = 6 sqrt(8) x
    # implies hessian = 6 sqrt(8) y * 18 sqrt(8) y - (6 sqrt(8) x)^2
    #                 = 108 * 8 * y^2 - 36 * 8 x^2 = 864 y^2 - 288 x^2
    # That's a little inconvenient to decompose into Zernikes by hand, but we can test against
    # an array of (x,y) values.
    np.testing.assert_allclose(
        Zernike([0,0,0,0,0,0,0,1]).hessian(x, y),
        864*y*y - 288*x*x
    )


@timer
def test_describe_zernike():
    """Test that Zernike descriptions make sense."""
    # Just do a few by hand
    # These can be looked up in Lakshminarayanan & Fleck (2011), Journal of Modern Optics
    # Table 1 there has algebraic expressions for Zernikes through j=36
    # Note, their definition is slightly different than ours: x and y are swapped.  (See their
    # figure 2 in which the azimuthal angle is defined +ve CW from +y.  We use +ve CCW from +x to be
    # consistent with Zemax.)
    assert galsim.zernike.describe_zernike(1) == "sqrt(1) * (1)"
    assert galsim.zernike.describe_zernike(2) == "sqrt(4) * (x)"
    assert galsim.zernike.describe_zernike(3) == "sqrt(4) * (y)"
    assert galsim.zernike.describe_zernike(4) == "sqrt(3) * (-1 + 2y^2 + 2x^2)"
    assert galsim.zernike.describe_zernike(10) == "sqrt(8) * (-3xy^2 + x^3)"

    Z22str = (
        "sqrt(7) * (-1 + 12y^2 - 30y^4 + 20y^6 + 12x^2"
        " - 60x^2y^2 + 60x^2y^4 - 30x^4 + 60x^4y^2 + 20x^6)"
    )
    assert galsim.zernike.describe_zernike(22) == Z22str

    Z36str = (
        "sqrt(16) * (-7xy^6 + 35x^3y^4 - 21x^5y^2 + x^7)"
    )
    assert galsim.zernike.describe_zernike(36) == Z36str


@timer
def test_lazy_coef():
    """Check that coefs reconstructed from _coef_array_xy round trip correctly."""
    bd = galsim.BaseDeviate(191120)
    u = galsim.UniformDeviate(bd)
    # For triangular jmax, get the same shape array back.
    for jmax in [3, 6, 10, 15, 21]:
        zarr = [0]+[u() for i in range(jmax)]
        R_inner = u()*0.5+0.2
        R_outer = u()*2.0+2.0
        Z = Zernike(zarr, R_outer=R_outer, R_inner=R_inner)
        Z._coef_array_xy
        del Z.coef
        np.testing.assert_allclose(zarr, Z.coef, rtol=0, atol=1e-12)

    # For non-triangular jmax, get shape rounded up to next triangular
    for jmax in [2, 7, 11, 17, 23]:
        zarr = [0]+[u() for i in range(jmax)]
        R_inner = u()*0.5+0.2
        R_outer = u()*2.0+2.0
        Z = Zernike(zarr, R_outer=R_outer, R_inner=R_inner)
        Z._coef_array_xy
        del Z.coef
        np.testing.assert_allclose(zarr, Z.coef[:len(zarr)], rtol=0, atol=1e-12)
        # extra coefficients are all ~0
        np.testing.assert_allclose(Z.coef[len(zarr):], 0.0, rtol=0, atol=1e-12)


@timer
def test_dz_val():
    rng = galsim.BaseDeviate(1234).as_numpy_generator()
    for _ in range(10):
        kmax = rng.integers(4, 12)
        jmax = rng.integers(4, 12)
        coef = rng.normal(size=(kmax+1, jmax+1))
        uv_inner = rng.uniform(0.4, 0.7)
        uv_outer = rng.uniform(1.3, 1.7)
        xy_inner = rng.uniform(0.4, 0.7)
        xy_outer = rng.uniform(1.3, 1.7)
        dz = DoubleZernike(
            coef,
            uv_inner=uv_inner,
            uv_outer=uv_outer,
            xy_inner=xy_inner,
            xy_outer=xy_outer,
        )

        uv_scalar = rng.normal(size=(2,))
        xy_scalar = rng.normal(size=(2,))
        uv_vector = rng.normal(size=(2, 10))
        xy_vector = rng.normal(size=(2, 10))

        check_pickle(dz)
        check_pickle(dz, lambda dz_: dz_.coef.shape)
        check_pickle(dz, lambda dz_: tuple(dz_.coef.ravel()))
        check_pickle(dz, lambda dz_: dz_._coef_array_uvxy.shape)
        check_pickle(dz, lambda dz_: tuple(dz_._coef_array_uvxy.ravel()))
        check_pickle(dz, lambda dz_: dz_(*uv_scalar))
        check_pickle(dz, lambda dz_: tuple(dz_(*uv_vector)))
        check_pickle(dz, lambda dz_: dz_(*uv_scalar, *xy_scalar))
        check_pickle(dz, lambda dz_: tuple(dz_(*uv_vector, *xy_vector)))

        # If you don't specify xy, then get (list of) Zernike out.
        assert isinstance(dz(*uv_scalar), Zernike)
        assert isinstance(dz(*uv_vector), list)
        assert all(isinstance(z, Zernike) for z in dz(*uv_vector))

        # If uv scalar and xy scalar, then get scalar out.
        assert np.ndim(dz(*uv_scalar, *xy_scalar)) == 0
        # If uv vector and xy scalar, then get vector out.
        assert np.ndim(dz(*uv_vector, *xy_scalar)) == 1
        # If uv scalar and xy vector, then get vector out.
        assert np.ndim(dz(*uv_scalar, *xy_vector)) == 1
        # If uv vector and xy vector, then get vector out.
        assert np.ndim(dz(*uv_vector, *xy_vector)) == 1

        # Check consistency of __call__ outputs
        zk_list = dz(*uv_vector)
        vals = dz(*uv_vector, *xy_vector)
        np.testing.assert_allclose(
            np.array([zk(x, y) for x, y, zk in zip(*xy_vector, zk_list)]),
            vals,
            atol=2e-13, rtol=0
        )
        for i, (x, y) in enumerate(xy_vector.T):
            np.testing.assert_allclose(
                vals[i],
                zk_list[i](x, y),
                atol=2e-13, rtol=0
            )
        for i, (u, v, x, y) in enumerate(zip(*uv_vector, *xy_vector)):
            np.testing.assert_allclose(
                vals[i],
                dz(u, v, x, y),
                atol=2e-13, rtol=0
            )
        for i, (u, v) in enumerate(zip(*uv_vector)):
            np.testing.assert_allclose(
                vals[i],
                dz(u, v)(*xy_vector[:, i]),
                atol=2e-13, rtol=0
            )

        # Check asserts
        with assert_raises(AssertionError):
            dz(0.0, [1.0])
        with assert_raises(AssertionError):
            dz([0.0], [1.0, 1.0])
        with assert_raises(galsim.GalSimIncompatibleValuesError):
            dz(0.0, 0.0, x=0.0, y=None)
        with assert_raises(AssertionError):
            dz(0.0, 0.0, x=[1.0], y=1.0)
        with assert_raises(AssertionError):
            dz(0.0, 0.0, x=[1.0], y=[1.0, 2.0])
        with assert_raises(AssertionError):
            dz([0.0, 1.0], [0.0, 1.0], x=[1.0], y=[1.0])

    # Try pickle/repr with default domain
    dz = DoubleZernike(coef)
    check_pickle(dz)


@timer
def test_dz_coef_uvxy():
    rng = galsim.BaseDeviate(4321).as_numpy_generator()
    for _ in range(100):
        kmax = rng.integers(4, 22)
        jmax = rng.integers(4, 22)
        coef = rng.normal(size=(kmax+1, jmax+1))
        coef[0] = 0.0
        coef[:, 0] = 0.0
        uv_inner = rng.uniform(0.4, 0.7)
        uv_outer = rng.uniform(1.3, 1.7)
        xy_inner = rng.uniform(0.4, 0.7)
        xy_outer = rng.uniform(1.3, 1.7)
        dz = DoubleZernike(
            coef,
            uv_inner=uv_inner,
            uv_outer=uv_outer,
            xy_inner=xy_inner,
            xy_outer=xy_outer
        )
        # Test that we can recover coef from coef_array_xyuv
        dz._coef_array_uvxy
        del dz.coef
        np.testing.assert_allclose(
            dz.coef[:coef.shape[0], :coef.shape[1]],
            coef,
            rtol=0,
            atol=1e-12
        )

        uv_scalar = rng.normal(size=(2,))
        xy_scalar = rng.normal(size=(2,))
        uv_vector = rng.normal(size=(2, 10))
        xy_vector = rng.normal(size=(2, 10))

        # Scalar uv only
        zk1 = dz._call_old(*uv_scalar)
        zk2 = dz(*uv_scalar)
        n = len(zk1.coef)
        np.testing.assert_allclose(
            zk1.coef[1:n],
            zk2.coef[1:n],
            rtol=1e-11, atol=1e-11
        )

        # Vector uv only
        zks1 = dz._call_old(*uv_vector)
        zks2 = dz(*uv_vector)
        for zk1, zk2 in zip(zks1, zks2):
            n = len(zk1.coef)
            np.testing.assert_allclose(
                zk1.coef[1:n],
                zk2.coef[1:n],
                rtol=1e-11, atol=1e-11
            )

        # All scalar/vector combinations
        for uv in [uv_scalar, uv_vector]:
            for xy in [xy_scalar, xy_vector]:
                np.testing.assert_allclose(
                    dz(*uv, *xy),
                    dz._call_old(*uv, *xy)
                )


@timer
def test_dz_sum():
    """Test that DZ.__add__, __sub__, and __neg__ work as expected.
    """
    rng = galsim.BaseDeviate(57721).as_numpy_generator()
    u = rng.uniform(-1.0, 1.0, size=100)
    v = rng.uniform(-1.0, 1.0, size=100)
    x = rng.uniform(-1.0, 1.0, size=100)
    y = rng.uniform(-1.0, 1.0, size=100)

    for _ in range(100):
        k1 = rng.integers(1, 11)
        j1 = rng.integers(1, 11)
        k2 = rng.integers(1, 11)
        j2 = rng.integers(1, 11)

        uv_inner = rng.uniform(0.4, 0.7)
        uv_outer = rng.uniform(1.3, 1.7)
        xy_inner = rng.uniform(0.4, 0.7)
        xy_outer = rng.uniform(1.3, 1.7)

        coef1 = rng.normal(size=(k1+1, j1+1))
        coef1[0] = 0.0
        coef1[:, 0] = 0.0
        coef2 = rng.normal(size=(k2+1, j2+1))
        coef2[0] = 0.0
        coef2[:, 0] = 0.0

        dz1 = DoubleZernike(
            coef1,
            uv_inner=uv_inner, uv_outer=uv_outer,
            xy_inner=xy_inner, xy_outer=xy_outer
        )
        dz2 = DoubleZernike(
            coef2,
            uv_inner=uv_inner, uv_outer=uv_outer,
            xy_inner=xy_inner, xy_outer=xy_outer
        )
        c1 = rng.uniform(-1.0, 1.0)
        c2 = rng.uniform(-1.0, 1.0)

        kmax = max(k1, k2)
        jmax = max(j1, j2)

        coefSum = np.zeros((kmax+1, jmax+1))
        coefSum[:k1+1, :j1+1] = c1*coef1
        coefSum[:k2+1, :j2+1] += c2*coef2

        coefDiff = np.zeros((kmax+1, jmax+1))
        coefDiff[:k1+1, :j1+1] = c1*coef1
        coefDiff[:k2+1, :j2+1] -= c2*coef2

        np.testing.assert_allclose(coefSum, (c1*dz1 + c2*dz2).coef)
        np.testing.assert_allclose(coefDiff, (c1*dz1 - c2*dz2).coef)

        np.testing.assert_allclose(
            c1*dz1(u, v, x, y) + c2*dz2(u, v, x, y),
            (c1*dz1 + c2*dz2)(u, v, x, y)
        )
        np.testing.assert_allclose(
            c1*dz1(u, v, x, y) - c2*dz2(u, v, x, y),
            (c1*dz1 - c2*dz2)(u, v, x, y)
        )

        # Check that domains are preserved
        dzsum = dz1 + dz2
        np.testing.assert_allclose(
            dzsum.uv_inner,
            uv_inner
        )
        np.testing.assert_allclose(
            dzsum.uv_outer,
            uv_outer
        )
        np.testing.assert_allclose(
            dzsum.xy_inner,
            xy_inner
        )
        np.testing.assert_allclose(
            dzsum.xy_outer,
            xy_outer
        )

        with np.testing.assert_raises(TypeError):
            dz1 + 1
        with np.testing.assert_raises(TypeError):
            dz1 - 3
        with np.testing.assert_raises(ValueError):
            dz1 + DoubleZernike(
                coef1, uv_outer=2*uv_outer, uv_inner=uv_inner, xy_outer=xy_outer, xy_inner=xy_inner
            )
        with np.testing.assert_raises(ValueError):
            dz1 + DoubleZernike(
                coef1, uv_outer=uv_outer, uv_inner=2*uv_inner, xy_outer=xy_outer, xy_inner=xy_inner
            )
        with np.testing.assert_raises(ValueError):
            dz1 + DoubleZernike(
                coef1, uv_outer=uv_outer, uv_inner=uv_inner, xy_outer=2*xy_outer, xy_inner=xy_inner
            )
        with np.testing.assert_raises(ValueError):
            dz1 + DoubleZernike(
                coef1, uv_outer=uv_outer, uv_inner=uv_inner, xy_outer=xy_outer, xy_inner=2*xy_inner
            )

        # Commutative with integer coefficients
        dz1 = DoubleZernike(np.eye(3, dtype=int))
        dz2 = DoubleZernike(np.ones((4, 4), dtype=int))
        assert dz1 + dz2 == dz2 + dz1
        assert (dz2 - dz1) == dz2 + (-dz1) == -(dz1 - dz2)

        # Check again with missing .coef
        del dz1.coef
        del dz2.coef
        assert dz1 + dz2 == dz2 + dz1
        assert (dz2 - dz1) == dz2 + (-dz1) == -(dz1 - dz2)


@timer
def test_dz_product():
    """Test that __mul__ and __rmul__ work as expected.
    """
    rng = galsim.BaseDeviate(31415).as_numpy_generator()
    u = rng.uniform(-1.0, 1.0, size=100)
    v = rng.uniform(-1.0, 1.0, size=100)
    x = rng.uniform(-1.0, 1.0, size=100)
    y = rng.uniform(-1.0, 1.0, size=100)

    for _ in range(100):
        k1 = rng.integers(1, 16)
        j1 = rng.integers(1, 16)
        k2 = rng.integers(1, 16)
        j2 = rng.integers(1, 16)

        uv_inner = rng.uniform(0.4, 0.7)
        uv_outer = rng.uniform(1.3, 1.7)
        xy_inner = rng.uniform(0.4, 0.7)
        xy_outer = rng.uniform(1.3, 1.7)

        coef1 = rng.normal(size=(k1+1, j1+1))
        coef1[0] = 0.0
        coef1[:, 0] = 0.0
        coef2 = rng.normal(size=(k2+1, j2+1))
        coef2[0] = 0.0
        coef2[:, 0] = 0.0

        dz1 = DoubleZernike(
            coef1,
            uv_inner=uv_inner, uv_outer=uv_outer,
            xy_inner=xy_inner, xy_outer=xy_outer
        )
        dz2 = DoubleZernike(
            coef2,
            uv_inner=uv_inner, uv_outer=uv_outer,
            xy_inner=xy_inner, xy_outer=xy_outer
        )

        np.testing.assert_allclose(
            dz1(u, v, x, y)*dz2(u, v, x, y),
            (dz1 * dz2)(u, v, x, y)
        )
        np.testing.assert_allclose(
            dz1(u, v, x, y)*dz2(u, v, x, y),
            (dz2 * dz1)(u, v, x, y)
        )
        # Check scalar multiplication
        np.testing.assert_allclose(
            2*dz1(u, v, x, y),
            (2 * dz1)(u, v, x, y)
        )
        np.testing.assert_allclose(
            dz1(u, v, x, y)*3.3,
            (dz1 * 3.3)(u, v, x, y)
        )
        # Try when .coef is missing
        del dz1.coef
        np.testing.assert_allclose(
            dz1(u, v, x, y)*3.5,
            (dz1 * 3.5)(u, v, x, y)
        )
        # Check that domain is preserved
        dzprod = dz1 * dz2
        np.testing.assert_equal(
            [dzprod.uv_inner, dzprod.uv_outer, dzprod.xy_inner, dzprod.xy_outer],
            [uv_inner, uv_outer, xy_inner, xy_outer]
        )
        # Check div
        np.testing.assert_allclose(
            (dz1 / 5.6)(u, v, x, y),
            dz1(u, v, x, y)/5.6,
        )


    with np.testing.assert_raises(TypeError):
        dz1 * galsim.Gaussian(sigma=1.0)
    with np.testing.assert_raises(ValueError):
        dz1 * DoubleZernike(
            coef1, uv_outer=2*uv_outer, uv_inner=uv_inner, xy_outer=xy_outer, xy_inner=xy_inner
        )
    with np.testing.assert_raises(ValueError):
        dz1 * DoubleZernike(
            coef1, uv_outer=uv_outer, uv_inner=2*uv_inner, xy_outer=xy_outer, xy_inner=xy_inner
        )
    with np.testing.assert_raises(ValueError):
        dz1 * DoubleZernike(
            coef1, uv_outer=uv_outer, uv_inner=uv_inner, xy_outer=2*xy_outer, xy_inner=xy_inner
        )
    with np.testing.assert_raises(ValueError):
        dz1 * DoubleZernike(
            coef1, uv_outer=uv_outer, uv_inner=uv_inner, xy_outer=xy_outer, xy_inner=2*xy_inner
        )
    with np.testing.assert_raises(TypeError):
        dz1 / dz2

    # Commutative with integer coefficients
    dz1 = DoubleZernike(np.eye(3, dtype=int))
    dz2 = DoubleZernike(np.ones((4, 4), dtype=int))
    assert dz1 * dz2 == dz2 * dz1
    assert (dz2 * 3) == (3 * dz2)


@timer
def test_dz_grad():
    """Test that DZ gradients work as expected.
    """
    rng = galsim.BaseDeviate(31415).as_numpy_generator()
    u = rng.uniform(-1.0, 1.0, size=100)
    v = rng.uniform(-1.0, 1.0, size=100)
    x = rng.uniform(-1.0, 1.0, size=100)
    y = rng.uniform(-1.0, 1.0, size=100)

    for _ in range(10):
        k1 = rng.integers(1, 16)
        j1 = rng.integers(1, 16)

        uv_inner = rng.uniform(0.4, 0.7)
        uv_outer = rng.uniform(1.3, 1.7)
        xy_inner = rng.uniform(0.4, 0.7)
        xy_outer = rng.uniform(1.3, 1.7)

        coef = rng.normal(size=(k1+1, j1+1))
        coef[0] = 0.0
        coef[:, 0] = 0.0

        dz = DoubleZernike(
            coef,
            uv_inner=uv_inner, uv_outer=uv_outer,
            xy_inner=xy_inner, xy_outer=xy_outer
        )

        # X and Y are easy to check with single Zernike gradient functions.
        np.testing.assert_allclose(
            dz.gradX(u, v, x, y),
            np.array([dz(u_, v_).gradX(x_, y_) for u_, v_, x_, y_ in zip(u, v, x, y)])
        )
        np.testing.assert_allclose(
            dz.gradY(u, v, x, y),
            np.array([dz(u_, v_).gradY(x_, y_) for u_, v_, x_, y_ in zip(u, v, x, y)])
        )
        # U and V are trickier, since we aren't including a way to turn a DZ evaluated
        # at (x, y) into a single Zernike of (u, v).  We can mock that though up by
        # transposing the DZ coefficients and swapping the domain parameters.
        dz_xyuv = DoubleZernike(
            np.transpose(coef, axes=(1, 0)),
            uv_inner=xy_inner, uv_outer=xy_outer,
            xy_inner=uv_inner, xy_outer=uv_outer
        )
        np.testing.assert_allclose(
            dz.gradU(u, v, x, y),
            np.array([dz_xyuv(x_, y_).gradX(u_, v_) for u_, v_, x_, y_ in zip(u, v, x, y)])
        )
        np.testing.assert_allclose(
            dz.gradV(u, v, x, y),
            np.array([dz_xyuv(x_, y_).gradY(u_, v_) for u_, v_, x_, y_ in zip(u, v, x, y)])
        )

        # Test Zernike coefficients themselves.
        for u_, v_ in zip(u, v):
            dzdx1 = dz(u_, v_).gradX.coef
            dzdx2 = dz.gradX(u_, v_).coef
            dzdx1 = np.concatenate([dzdx1, np.zeros(len(dzdx2) - len(dzdx1))])
            np.testing.assert_allclose(dzdx1, dzdx2, atol=1e-12)

            dzdy1 = dz(u_, v_).gradY.coef
            dzdy2 = dz.gradY(u_, v_).coef
            dzdy1 = np.concatenate([dzdy1, np.zeros(len(dzdy2) - len(dzdy1))])
            np.testing.assert_allclose(dzdy1, dzdy2, atol=1e-12)


@timer
def test_dz_to_T():
    """Test that DZs enable efficient computation of optical PSF sizes."""

    # One of the reasons to build the DZ class was to see if we can quickly
    # convert from coefficients that, say, come out of batoid or Piff or an
    # active optics framework into a PSF size, both averaged and over the field
    # of view.

    # The idea is that x and y deflections of photons from their optimal paths
    # are directly proportional to the wavefront gradient.  I.e.,
    # delta_xfp \propto dW(u, v, x, y)/dx
    # where delta_xfp is the focal plane deflection, and W is the double Zernike
    # wavefront which is a function of field angle (u, v) and pupil position
    # (x, y).

    # We want, as a function of (u, v), the variance of delta_xfp.  This is
    # Var[delta_xfp] \propto Var[dW/dx] = E[(dW/dx)^2] - (E[dW/dx])^2,  where E
    # is the expectation value over the pupil.

    # First construct a DZ.
    rng = galsim.BaseDeviate(51413).as_numpy_generator()

    for _ in range(10):
        k1 = rng.integers(1, 29)
        j1 = rng.integers(4, 11)

        uv_inner = rng.uniform(0.4, 0.7)
        uv_outer = rng.uniform(1.3, 1.7)
        xy_inner = rng.uniform(0.4, 0.7)
        xy_outer = rng.uniform(1.3, 1.7)

        coef = rng.normal(size=(k1+1, j1+1))
        coef[0] = 0.0
        coef[:, 0] = 0.0

        W = DoubleZernike(
            coef,
            uv_inner=uv_inner, uv_outer=uv_outer,  # field
            xy_inner=xy_inner, xy_outer=xy_outer   # pupil
        )

        # Now the analytic map of optical PSF size T
        dWdx = W.gradX
        dWdy = W.gradY
        dWdx2 = dWdx * dWdx
        dWdy2 = dWdy * dWdy

        # These are still double Zernikes.  Extract their expectation values
        # over the pupil (which are functions of field angle).
        dWdx_field = dWdx.mean_xy
        dWdy_field = dWdy.mean_xy
        dWdx2_field = dWdx2.mean_xy
        dWdy2_field = dWdy2.mean_xy
        # Now construct the PSF size T
        T = dWdx2_field + dWdy2_field - dWdx_field*dWdx_field - dWdy_field*dWdy_field

        # Now compare to a quick Monte Carlo estimate of T.
        us = []
        vs = []
        Ts = []
        for _ in range(100):  # 100 positions
            # Pick a random field angle:
            uv = np.inf
            while uv > uv_outer or uv < uv_inner:
                u_ = rng.uniform(-uv_outer, uv_outer)
                v_ = rng.uniform(-uv_outer, uv_outer)
                uv = np.hypot(u_, v_)
            us.append(u_)
            vs.append(v_)
            # Get the wavefront at that field angle
            W_ = W(u_, v_)
            # Pick random pupil positions
            dxs = []
            dys = []
            for _ in range(50):  # 50 photons
                xy = np.inf
                while xy > xy_outer or xy < xy_inner:
                    x_ = rng.uniform(-xy_outer, xy_outer)
                    y_ = rng.uniform(-xy_outer, xy_outer)
                    xy = np.hypot(x_, y_)
                # Get the focal plane deflection
                dxs.append(W_.gradX(x_, y_))
                dys.append(W_.gradY(x_, y_))
            # Compute the variance of the focal plane deflection
            Ts.append(np.var(dxs) + np.var(dys))

        # Assert that the median relative error is less than 5%,
        # and the 90% worst case is less than 35%
        np.testing.assert_array_less(
            np.abs(np.quantile((T(us, vs) - Ts)/Ts, [0.5, 0.9])),
            [0.05, 0.35]
        )

        # Uncomment below to look at the plots

        # def colorbar(mappable):
        #     from mpl_toolkits.axes_grid1 import make_axes_locatable
        #     import matplotlib.pyplot as plt
        #     last_axes = plt.gca()
        #     ax = mappable.axes
        #     fig = ax.figure
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     cbar = fig.colorbar(mappable, cax=cax)
        #     plt.sca(last_axes)
        #     return cbar

        # ugrid = np.linspace(-uv_outer, uv_outer, 100)
        # ugrid, vgrid = np.meshgrid(ugrid, ugrid)
        # Tgrid = T(ugrid, vgrid)
        # w = np.hypot(ugrid, vgrid) > uv_outer
        # w |= np.hypot(ugrid, vgrid) < uv_inner
        # Tgrid[w] = np.nan

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
        # vmin, vmax = np.nanquantile(Tgrid, [0.02, 0.98])
        # colorbar(axes[0].scatter(us, vs, c=Ts, vmin=vmin, vmax=vmax))
        # colorbar(axes[1].imshow(Tgrid, extent=(-uv_outer, uv_outer, -uv_outer, uv_outer), origin='lower', vmin=vmin, vmax=vmax))
        # axes[1].scatter(us, vs, c=Ts, vmin=vmin, vmax=vmax)
        # axes[0].set_aspect('equal')
        # fig.tight_layout()
        # plt.show()


@timer
def test_dz_rotate():
    rng = galsim.BaseDeviate(12775).as_numpy_generator()

    for _ in range(30):
        k1 = rng.choice([1, 3, 10, 11, 13, 21, 22, 34])
        j1 = rng.choice([1, 3, 10, 11, 13, 21, 22, 34])

        uv_inner = rng.uniform(0.4, 0.7)
        uv_outer = rng.uniform(1.3, 1.7)
        xy_inner = rng.uniform(0.4, 0.7)
        xy_outer = rng.uniform(1.3, 1.7)

        coef = rng.normal(size=(k1+1, j1+1))
        coef[0] = 0.0
        coef[:, 0] = 0.0

        dz = DoubleZernike(
            coef,
            uv_inner=uv_inner, uv_outer=uv_outer,  # field
            xy_inner=xy_inner, xy_outer=xy_outer   # pupil
        )

        # Test points
        r = rng.uniform(0.0, 1.0, size=30)
        th = rng.uniform(0.0, 2*np.pi, size=30)
        rho = rng.uniform(0.0, 1.0, size=30)
        ph = rng.uniform(0.0, 2*np.pi, size=30)

        for theta_uv in [0.0, 0.2, 0.4]:
            for theta_xy in [0.0, 0.1, 0.3]:
                dz_rot = dz.rotate(theta_uv=theta_uv, theta_xy=theta_xy)
                np.testing.assert_allclose(
                    dz(
                        rho*np.cos(ph), rho*np.sin(ph),
                        r*np.cos(th), r*np.sin(th)
                    ),
                    dz_rot(
                        rho*np.cos(ph+theta_uv), rho*np.sin(ph+theta_uv),
                        r*np.cos(th+theta_xy), r*np.sin(th+theta_xy)
                    ),
                    atol=1e-11, rtol=0
                )


@timer
def test_dz_basis():
    """Test the doubleZernikeBasis function"""

    rng = galsim.BaseDeviate(127750).as_numpy_generator()

    for _ in range(10):
        k1 = rng.choice([3, 7, 10, 15])
        j1 = rng.choice([3, 5, 7])

        uv_inner = rng.uniform(0.4, 0.7)
        uv_outer = rng.uniform(1.3, 1.7)
        xy_inner = rng.uniform(0.4, 0.7)
        xy_outer = rng.uniform(1.3, 1.7)

        for _ in range(10):
            # Test at some random points
            x, y, u, v = rng.uniform(-0.5, 0.5, size=(4, 1000))

            # dzBases will generate all basis vectors at once
            dzBases = galsim.zernike.doubleZernikeBasis(
                k1, j1, x, y, u, v,
                uv_inner=uv_inner, uv_outer=uv_outer,
                xy_inner=xy_inner, xy_outer=xy_outer,
            )

            for j in range(1, j1):
                for k in range(1, k1):
                    coef = np.zeros((k1, j1))
                    coef[k, j] = 1.0
                    DZ = DoubleZernike(
                        coef,
                        uv_inner=uv_inner, uv_outer=uv_outer,
                        xy_inner=xy_inner, xy_outer=xy_outer
                    )
                    DZbasis = DZ(x, y, u, v)
                    np.testing.assert_allclose(
                        DZbasis, dzBases[k, j],
                        atol=1e-11, rtol=0
                    )


@timer
def test_dz_mean():
    """Test the dz.mean_xy and .mean_uv properties"""
    rng = galsim.BaseDeviate(51413).as_numpy_generator()

    for _ in range(10):
        k1 = rng.choice([3, 7, 10, 15])
        j1 = rng.choice([3, 5, 7])

        uv_inner = rng.uniform(0.4, 0.7)
        uv_outer = rng.uniform(1.3, 1.7)
        xy_inner = rng.uniform(0.4, 0.7)
        xy_outer = rng.uniform(1.3, 1.7)

        coef = rng.normal(size=(k1+1, j1+1))
        coef[0] = 0.0
        coef[:, 0] = 0.0

        dz = DoubleZernike(
            coef,
            uv_inner=uv_inner, uv_outer=uv_outer,
            xy_inner=xy_inner, xy_outer=xy_outer
        )
        # We don't have a function that returns a Zernike over uv at a given xy
        # point, but we can mimic that by transposing xy an uv in a new
        # DoubleZernike object.
        dzT = DoubleZernike(
            coef.T,
            uv_inner=xy_inner, uv_outer=xy_outer,
            xy_inner=uv_inner, xy_outer=uv_outer
        )

        mean_xy = dz.mean_xy
        mean_uv = dz.mean_uv

        for _ in range(10):
            # Pick a random uv point
            uv = np.inf
            while uv > uv_outer or uv < uv_inner:
                u = rng.uniform(-uv_outer, uv_outer)
                v = rng.uniform(-uv_outer, uv_outer)
                uv = np.hypot(u, v)
            # Evaluate at that point
            zk = dz(u, v)
            np.testing.assert_allclose(
                zk.coef[1],
                mean_xy(u, v),
            )

        for _ in range(10):
            # Pick a random xy point
            xy = np.inf
            while xy > xy_outer or xy < xy_inner:
                x = rng.uniform(-xy_outer, xy_outer)
                y = rng.uniform(-xy_outer, xy_outer)
                xy = np.hypot(x, y)
            # Evaluate at that point
            zk = dzT(x, y)
            np.testing.assert_allclose(
                zk.coef[1],
                mean_uv(x, y),
            )

        # Make sure we can construct from ._coef_array_uvxy when .coef is
        # unavailable too.
        dz._coef_array_uvxy  # Ensure _coef_array_uvxy exists
        del dz.coef, dz.mean_uv, dz.mean_xy  # Clear out lazy_properties
        mean_xy2 = dz.mean_xy
        mean_uv2 = dz.mean_uv

        sh = min(len(mean_xy.coef), len(mean_xy2.coef))
        np.testing.assert_allclose(
            mean_xy.coef[:sh], mean_xy2.coef[:sh],
            atol=1e-14, rtol=0
        )
        np.testing.assert_equal(
            (mean_xy.R_inner, mean_xy.R_outer),
            (mean_xy2.R_inner, mean_xy2.R_outer),
        )


        sh = min(len(mean_uv.coef), len(mean_uv2.coef))
        np.testing.assert_allclose(
            mean_uv.coef[:sh], mean_uv2.coef[:sh],
            atol=1e-14, rtol=0
        )
        np.testing.assert_equal(
            (mean_uv.R_inner, mean_uv.R_outer),
            (mean_uv2.R_inner, mean_uv2.R_outer),
        )


if __name__ == "__main__":
    runtests(__file__)
