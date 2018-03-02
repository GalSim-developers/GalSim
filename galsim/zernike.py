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
"""@file zernike.py
Module contains code for evaluating and fitting Zernike polynomials
"""

import numpy as np

import galsim

# Some utilities for working with Zernike polynomials
# Combinations.  n choose r.
# See http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
# This is J. F. Sebastian's answer.
def _nCr(n, r):
    if 0 <= r <= n:
        ntok = 1
        rtok = 1
        for t in range(1, min(r, n - r) + 1):
            ntok *= n
            rtok *= t
            n -= 1
        return ntok // rtok
    else:
        return 0


# Start off with the Zernikes up to j=15
_noll_n = [0,0,1,1,2,2,2,3,3,3,3,4,4,4,4,4]
_noll_m = [0,0,1,-1,0,-2,2,-1,1,-3,3,0,2,-2,4,-4]
def _noll_to_zern(j):
    """
    Convert linear Noll index to tuple of Zernike indices.
    j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike
    index.
    @param [in] j Zernike mode Noll index
    @return (n, m) tuple of Zernike indices
    @see <https://oeis.org/A176988>.
    """
    while len(_noll_n) <= j:
        n = _noll_n[-1] + 1
        _noll_n.extend( [n] * (n+1) )
        if n % 2 == 0:
            _noll_m.append(0)
            m = 2
        else:
            m = 1
        # pm = +1 if m values go + then - in pairs.
        # pm = -1 if m values go - then + in pairs.
        pm = +1 if (n//2) % 2 == 0 else -1
        while m <= n:
            _noll_m.extend([ pm * m , -pm * m ])
            m += 2

    return _noll_n[j], _noll_m[j]


def _zern_norm(n, m):
    r"""Normalization coefficient for zernike (n, m).

    Defined such that \int_{unit disc} Z(n1, m1) Z(n2, m2) dA = \pi if n1==n2 and m1==m2 else 0.0
    """
    if m == 0:
        return np.sqrt(1./(n+1))
    else:
        return np.sqrt(1./(2.*n+2))


def _zern_rho_coefs(n, m):
    """Compute coefficients of radial part of Zernike (n, m).
    """
    kmax = (n-abs(m)) // 2
    A = [0]*(n+1)
    val = _nCr(n,kmax) # The value for k = 0 in the equation below.
    for k in range(kmax):
        # val = (-1)**k * _nCr(n-k, k) * _nCr(n-2*k, kmax-k) / _zern_norm(n, m)
        # The above formula is faster as a recurrence relation:
        A[n-2*k] = val
        # Don't use *= since the factor is not an integer, but the result is.
        val = -val * (kmax-k)*(n-kmax-k) // ((n-k)*(k+1))
    A[n-2*kmax] = val
    return A


def _zern_coef_array(n, m, obscuration, shape):
    """Assemble coefficient array for evaluating Zernike (n, m) as the real part of a
    bivariate polynomial in abs(rho)^2 and rho, where rho is a complex array indicating position on
    a unit disc.

    @param n            Zernike radial coefficient
    @param m            Zernike azimuthal coefficient
    @param obscuration  Linear obscuration fraction.
    @param shape        Output array shape

    @returns    2D array of coefficients in |r|^2 and r, where r = u + 1j * v, and u, v are unit
                disk coordinates.
    """
    if shape is None:
        shape = ((n//2)+1, abs(m)+1)
    out = np.zeros(shape, dtype=np.complex128)
    if 0 < obscuration < 1:
        coefs = np.array(_annular_zern_rho_coefs(n, m, obscuration), dtype=np.complex128)
    elif obscuration == 0:
        coefs = np.array(_zern_rho_coefs(n, m), dtype=np.complex128)
    else:
        raise ValueError("Illegal obscuration: {}".format(obscuration))
    coefs /= _zern_norm(n, m)
    if m < 0:
        coefs *= -1j
    for i, c in enumerate(coefs[abs(m)::2]):
        out[i, abs(m)] = c
    return out


def __noll_coef_array(jmax, obscuration):
    """Assemble coefficient array for evaluating Zernike (n, m) as the real part of a
    bivariate polynomial in abs(rho)^2 and rho, where rho is a complex array indicating position on
    a unit disc.

    @param jmax         Maximum Noll coefficient
    @param obscuration  Linear obscuration fraction.

    @returns    2D array of coefficients in |r|^2 and r, where r = u + 1j * v, and u, v are unit
                disk coordinates.
    """
    maxn = _noll_to_zern(jmax)[0]
    shape = (maxn//2+1, maxn+1, jmax)  # (max power of |rho|^2,  max power of rho, noll index-1)
    shape1 = (maxn//2+1, maxn+1)

    out = np.zeros(shape, dtype=np.complex128)
    for j in range(1,jmax+1):
        n,m = _noll_to_zern(j)
        coef = _zern_coef_array(n,m,obscuration,shape1)
        out[:,:,j-1] = coef
    return out
_noll_coef_array = galsim.utilities.LRU_Cache(__noll_coef_array)


# Following 3 functions from
#
# "Zernike annular polynomials for imaging systems with annular pupils"
# Mahajan (1981) JOSA Vol. 71, No. 1.

# Mahajan's h-function normalization for annular Zernike coefficients.
def __h(m, j, eps):
    if m == 0:  # Equation (A5)
        return (1-eps**2)/(2*(2*j+1))
    else:  # Equation (A14)
        num = -(2*(2*j+2*m-1)) * _Q(m-1, j+1, eps)[0]
        den = (j+m)*(1-eps**2) * _Q(m-1, j, eps)[0]
        return num/den * _h(m-1, j, eps)
_h = galsim.utilities.LRU_Cache(__h)


# Mahajan's Q-function for annular Zernikes.
def __Q(m, j, eps):
    if m == 0:  # Equation (A4)
        return _annular_zern_rho_coefs(2*j, 0, eps)[::2]
    else:  # Equation (A13)
        num = 2*(2*j+2*m-1) * _h(m-1, j, eps)
        den = (j+m)*(1-eps**2)*_Q(m-1, j, eps)[0]
        summation = np.zeros((j+1,), dtype=float)
        for i in range(j+1):
            qq = _Q(m-1, i, eps)
            qq = qq*qq[0]  # Don't use *= here since it modifies the cache!
            summation[:i+1] += qq/_h(m-1, i, eps)
        return summation * num / den
_Q = galsim.utilities.LRU_Cache(__Q)


def __annular_zern_rho_coefs(n, m, eps):
    """Compute coefficients of radial part of annular Zernike (n, m), with fractional linear
    obscuration eps.
    """
    out = np.zeros((n+1,), dtype=float)
    m = abs(m)
    if m == 0:  # Equation (18)
        norm = 1./(1-eps**2)
        # R[n, m=0, eps](r^2) = R[n, m=0, eps=0]((r^2 - eps^2)/(1 - eps^2))
        # Implement this by retrieving R[n, 0] coefficients of (r^2)^k and
        # multiplying in the binomial (in r^2) expansion of ((r^2 - eps^2)/(1 - eps^2))^k
        coefs = _zern_rho_coefs(n, 0)
        for i, coef in enumerate(coefs):
            if i % 2 == 1: continue
            j = i // 2
            more_coefs = (norm**j) * galsim.utilities.binomial(-eps**2, 1, j)
            out[0:i+1:2] += coef*more_coefs
    elif m == n:  # Equation (25)
        norm = 1./np.sqrt(np.sum((eps**2)**np.arange(n+1)))
        out[n] = norm
    else:  # Equation (A1)
        j = (n-m)//2
        norm = np.sqrt((1-eps**2)/(2*(2*j+m+1) * _h(m,j,eps)))
        out[m::2] = norm * _Q(m, j, eps)
    return out
_annular_zern_rho_coefs = galsim.utilities.LRU_Cache(__annular_zern_rho_coefs)


class Zernike(object):
    def __init__(self, a, eps=0.0, diam=2.0):
        """a[0] is piston, a[1] is tip, a[2] is tilt, a[3] is defocus, etc.
        So the traditional Z4 coefficient (defocus) is given by a[3]
        eps is fractional linear obscuration.  Implies use of annular Zernikes.
        """
        self.a = a
        self._jmax = len(self.a)
        self._nmax, _ = _noll_to_zern(self._jmax)
        shape = (self._nmax//2+1, self._nmax+1)  # (max power of |rho|^2, max power of |rho|)
        self._coef_array = np.zeros(shape, dtype=np.complex128)
        noll_coef = _noll_coef_array(self._jmax, eps)
        self._coef_array = np.dot(noll_coef, self.a)
        if diam != 2.0:
            self._coef_array /= (diam/2)**np.sum(np.mgrid[0:2*shape[0]:2, 0:shape[1]], axis=0)

    def evalCartesian(self, x, y):
        r = x + 1j * y
        rsqr = np.abs(r)**2
        return galsim.utilities.horner2d(rsqr, r, self._coef_array, dtype=complex).real

    def evalPolar(self, rho, theta):
        cth = np.cos(theta)
        sth = np.sin(theta)
        r = rho * cth + 1j * rho * sth
        rsqr = rho**2
        return galsim.utilities.horner2d(rsqr, r, self._coef_array, dtype=complex).real

    def rotate(self, theta):
        # Use formula from Tatulli (2013) arXiv:1302.7106v1

        # need shape jmax to be odd (even) when nmax is odd (even), so
        # append zeros to a until jmax+nmax is even.
        a = self.a
        if (self._nmax//2 + self._jmax) % 2 == 0:
            a = np.append(a, [0])
        M = np.zeros((len(a), len(a)), dtype=np.float64)
        for i in range(len(a)):
            ni, mi = _noll_to_zern(i+1)
            for j in range(len(a)):
                nj, mj = _noll_to_zern(j+1)
                if ni != nj:
                    continue
                if abs(mi) != abs(mj):
                    continue
                if mi == mj:
                    M[i, j] = np.cos(mj * theta)
                elif mi == -mj:
                    M[i, j] = np.sin(mj * theta)
        return Zernike(np.dot(M, a))
