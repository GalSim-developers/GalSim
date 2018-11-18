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
"""@file zernike.py
Module contains code for evaluating and fitting Zernike polynomials
"""

import numpy as np

from .utilities import LRU_Cache, binomial, horner2d, nCr, lazy_property
from .errors import GalSimValueError, GalSimRangeError

# Some utilities for working with Zernike polynomials

# Start off with the Zernikes up to j=15
_noll_n = [0,0,1,1,2,2,2,3,3,3,3,4,4,4,4,4]
_noll_m = [0,0,1,-1,0,-2,2,-1,1,-3,3,0,2,-2,4,-4]
def noll_to_zern(j):
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
    val = nCr(n,kmax) # The value for k = 0 in the equation below.
    for k in range(kmax):
        # val = (-1)**k * _nCr(n-k, k) * _nCr(n-2*k, kmax-k)
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
    out = np.zeros(shape, dtype=np.complex128)
    if 0 < obscuration < 1:
        coefs = np.array(_annular_zern_rho_coefs(n, m, obscuration), dtype=np.complex128)
    elif obscuration == 0:
        coefs = np.array(_zern_rho_coefs(n, m), dtype=np.complex128)
    else:
        raise GalSimRangeError("Invalid obscuration.", obscuration, 0., 1.)
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
    maxn = noll_to_zern(jmax)[0]
    shape = (maxn//2+1, maxn+1, jmax)  # (max power of |rho|^2,  max power of rho, noll index-1)

    out = np.zeros(shape, dtype=np.complex128)
    for j in range(1,jmax+1):
        n,m = noll_to_zern(j)
        coef = _zern_coef_array(n,m,obscuration,shape[0:2])
        out[:,:,j-1] = coef
    return out
_noll_coef_array = LRU_Cache(__noll_coef_array)


def _xy_contribution(rho2_power, rho_power, shape):
    """Convert (rho, |rho|^2) bivariate polynomial coefficients to (x, y) bivariate polynomial
    coefficients.
    """
    # rho = (x + iy), so multiplying an xy coefficient by rho, and again by rho is equivalent to:
    #
    # 1 0 0      0 i 0      0 0 -1
    # 0 0 0  ->  1 0 0  ->  0 2i 0
    # 0 0 0      0 0 0      1 0  0
    #
    # where the rows indicate powers of x and the columns indicate powers of y.
    # So the last array above indicates (x + iy)^2 = (x^2 + 2ixy - y^2)
    # Similarly, multiplying by |rho|^2 = x^2 + y^2 is equivalent to
    #
    # 1 0 0 0 0     0 0 1 0 0      0 0 0 0 1
    # 0 0 0 0 0     0 0 0 0 0      0 0 0 0 0
    # 0 0 0 0 0  -> 1 0 0 0 0  ->  0 0 2 0 0
    # 0 0 0 0 0     0 0 0 0 0      0 0 0 0 0
    # 0 0 0 0 0     0 0 0 0 0      1 0 0 0 0
    #
    # and so on.  We can apply these operations repeatedly to effect arbitrary powers of rho or
    # |rho|^2.
    out = np.zeros(shape, dtype=np.complex128)
    out[0,0] = 1
    while rho2_power >= 1:
        new = np.zeros_like(out)
        for (i, j) in zip(*np.nonzero(out)):
            val = out[i, j]
            new[i+2, j] += val
            new[i, j+2] += val
        rho2_power -= 1
        out = new
    while rho_power >= 1:
        new = np.zeros_like(out)
        for (i, j) in zip(*np.nonzero(out)):
            val = out[i, j]
            new[i+1, j] += val
            new[i, j+1] += 1j*val
        rho_power -= 1
        out = new
    return out


def _rrsq_to_xy(coefs, shape):
    """Convert coefficient array from rho, |rho|^2 to x, y.
    """
    new_coefs = np.zeros(shape, dtype=np.float64)

    # Now we loop through the elements of coefs and compute their contribution to new_coefs
    for (i, j) in zip(*np.nonzero(coefs)):
        new_coefs += (coefs[i, j]*_xy_contribution(i, j, shape)).real
    return new_coefs


def _xycoef_gradx(coefs, shape):
    """Calculate x/y coefficient array of x-derivative of given x/y coefficient array.
    """
    # d/dx (x+y) = 1 looks like
    #
    # 0 1      1 0
    # 1 0  ->  0 0
    #
    # d/dx (x^2 + y^2) = 2 x looks like
    #
    # 0 0 1    0 0 0
    # 0 0 0 -> 2 0 0
    # 1 0 0    0 0 0
    #
    # d/dx (x^2 + xy + y^2) = 2x + y looks like
    #
    # 0 0 1    0 1 0
    # 0 1 0 -> 2 0 0
    # 1 0 0    0 0 0

    gradx = np.zeros(shape, dtype=np.float64)
    for (i, j) in zip(*np.nonzero(coefs)):
        if i > 0:
            gradx[i-1, j] = coefs[i, j]*i

    return gradx


def _xycoef_grady(coefs, shape):
    """Calculate x/y coefficient array of y-derivative of given x/y coefficient array.
    """
    # see above
    grady = np.zeros(shape, dtype=np.float64)
    for (i, j) in zip(*np.nonzero(coefs)):
        if j > 0:
            grady[i, j-1] = coefs[i, j]*j

    return grady


def __noll_coef_array_xy(jmax, obscuration):
    """Assemble coefficient array for evaluating Zernike (n, m) as a bivariate polynomial in
    x and y.

    @param jmax         Maximum Noll coefficient
    @param obscuration  Linear obscuration fraction.

    @returns    2D array of coefficients in x and y.
    """
    maxn, _ = noll_to_zern(jmax)
    shape = (maxn+1, maxn+1, jmax)  # (max power of x, max power of y, noll index)

    nca = _noll_coef_array(jmax, obscuration)

    out = np.zeros(shape, dtype=np.float64)
    for j in range(1, jmax+1):
        out[:,:,j-1] = _rrsq_to_xy(nca[:,:,j-1], shape=shape[0:2])
    return out
_noll_coef_array_xy = LRU_Cache(__noll_coef_array_xy)


def __noll_coef_array_xy_gradx(jmax, obscuration):
    """Assemble coefficient array for evaluating the x-derivative of Zernike (n, m) as a bivariate
    polynomial in x and y.

    @param jmax         Maximum Noll coefficient
    @param obscuration  Linear obscuration fraction.

    @returns    2D array of coefficients in x and y.
    """
    maxn, _ = noll_to_zern(jmax)
    shape = (maxn+1, maxn+1, jmax)  # (max power of x, max power of y, noll index)

    nca = _noll_coef_array(jmax, obscuration)

    out = np.zeros(shape, dtype=np.float64)
    for j in range(1, jmax+1):
        out[:,:,j-1] = _xycoef_gradx(_rrsq_to_xy(nca[:,:,j-1], shape=shape[0:2]), shape=shape[0:2])
    return out[:-1, :-1, :]
_noll_coef_array_xy_gradx = LRU_Cache(__noll_coef_array_xy_gradx)


def __noll_coef_array_xy_grady(jmax, obscuration):
    """Assemble coefficient array for evaluating the y-derivative of Zernike (n, m) as a bivariate
    polynomial in x and y.

    @param jmax         Maximum Noll coefficient
    @param obscuration  Linear obscuration fraction.

    @returns    2D array of coefficients in x and y.
    """
    maxn = noll_to_zern(jmax)[0]
    shape = (maxn+1, maxn+1, jmax)  # (max power of x, max power of y, noll index-1)

    nca = _noll_coef_array(jmax, obscuration)

    out = np.zeros(shape, dtype=np.float64)
    for j in range(1, jmax+1):
        out[:,:,j-1] = _xycoef_grady(_rrsq_to_xy(nca[:,:,j-1], shape=shape[0:2]), shape=shape[0:2])
    return out[:-1, :-1, :]
_noll_coef_array_xy_grady = LRU_Cache(__noll_coef_array_xy_grady)


def __noll_coef_array_gradx(j, obscuration):
    if j == 1:
        return np.zeros((1,1), dtype=np.float)
    n, _ = noll_to_zern(j)
    # Gradient of Zernike with radial coefficient n has radial coefficient n-1.
    # Next line computes the largest j for which radial coefficient is n-1.
    jgrad = n*(n+1)//2
    # Solve for gradient coefficients in terms of non-gradient coefficients.
    return np.linalg.lstsq(
        _noll_coef_array_xy(jgrad, obscuration).reshape(-1, jgrad),
        _noll_coef_array_xy_gradx(j, obscuration).reshape(-1, j),
        rcond=-1.
    )[0]
_noll_coef_array_gradx = LRU_Cache(__noll_coef_array_gradx)


def __noll_coef_array_grady(j, obscuration):
    if j == 1:
        return np.zeros((1,1), dtype=np.float)
    n, _ = noll_to_zern(j)
    # Gradient of Zernike with radial coefficient n has radial coefficient n-1.
    # Next line computes the largest j for which radial coefficient is n-1.
    jgrad = n*(n+1)//2
    # Solve for gradient coefficients in terms of non-gradient coefficients.
    return np.linalg.lstsq(
        _noll_coef_array_xy(jgrad, obscuration).reshape(-1, jgrad),
        _noll_coef_array_xy_grady(j, obscuration).reshape(-1, j),
        rcond=-1.
    )[0]
_noll_coef_array_grady = LRU_Cache(__noll_coef_array_grady)


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
_h = LRU_Cache(__h)


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
_Q = LRU_Cache(__Q)


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
            more_coefs = (norm**j) * binomial(-eps**2, 1, j)
            out[0:i+1:2] += coef*more_coefs
    elif m == n:  # Equation (25)
        norm = 1./np.sqrt(np.sum((eps**2)**np.arange(n+1)))
        out[n] = norm
    else:  # Equation (A1)
        j = (n-m)//2
        norm = np.sqrt((1-eps**2)/(2*(2*j+m+1) * _h(m,j,eps)))
        out[m::2] = norm * _Q(m, j, eps)
    return out
_annular_zern_rho_coefs = LRU_Cache(__annular_zern_rho_coefs)


class Zernike(object):
    r"""A class to represent a Zernike polynomial series
    (http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials).

    Zernike polynomials form an orthonormal basis over the unit circle.  The convention used here is
    for the normality constant to equal the area of integration, which is pi for the unit circle.
    I.e.,

        \int_{unit circle} Z_i Z_j dA = \pi \delta_{i, j}.

    Two generalizations of the unit circle Zernike polynomials are also available in this class:
    annular Zernike polynomials, and polynomials defined over non-unit-radius circles.

    Annular Zernikes are orthonormal over an annulus instead of a circle (see Mahajan, J. Opt. Soc.
    Am. 71, 1, (1981)).  Similarly, the non-unit-radius polynomials are orthonormal over a region
    with outer radius not equal to 1.  Taken together, these generalizations yield the
    orthonormality condition:

        \int_{annulus} Z_i Z_j dA = \pi (R_outer^2 - R_inner^2) \delta_{i, j}

    where 0 <= R_inner < R_outer indicate the inner and outer radii of the annulus over which the
    polynomials are orthonormal.

    The indexing convention for i and j above is that from Noll, J. Opt. Soc. Am. 66, 207-211(1976).
    Note that the Noll indices begin at 1; there is no Z_0.  Because of this, the series
    coefficients argument `coef` effectively begins with `coef[1]` (`coef[0]` is ignored).  This
    convention is used consistently throughout GalSim, e.g., `OpticalPSF`, `OpticalScreen`,
    `zernikeRotMatrix`, and `zernikeBasis`.

    As an example, the first few Zernike polynomials in terms of Cartesian coordinates x and y are

        Noll index  |  polynomial
        --------------------------
             1      |      1
             2      |      2 x
             3      |      2 y
             4      |  sqrt(3) (2 (x^2 + y^2) - 1)

    @param coef     Zernike series coefficients.  Note that coef[i] corresponds to Z_i under the
                    Noll index convention, and coef[0] is ignored.  (I.e., coef[1] is 'piston',
                    coef[4] is 'defocus', ...)
    @param R_outer  Outer radius.  [default: 1.0]
    @param R_inner  Inner radius.  [default: 0.0]
    """
    def __init__(self, coef, R_outer=1.0, R_inner=0.0):
        self.coef = np.asarray(coef)
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)

    @lazy_property
    def _coef_array_xy(self):
        arr = _noll_coef_array_xy(len(self.coef)-1, self.R_inner/self.R_outer).dot(self.coef[1:])

        if self.R_outer != 1.0:
            n = arr.shape[0]
            norm = np.power(1./self.R_outer, np.arange(1,n))
            arr[0,1:] *= norm
            for i in range(1,n):
                arr[i,0:-i] *= norm[i-1:]
        return arr

    @lazy_property
    def gradX(self):
        """The x-derivative of this Zernike as a new Zernike object.
        """
        j = len(self.coef)-1
        ncagx = _noll_coef_array_gradx(j, self.R_inner/self.R_outer).dot(self.coef[1:])
        newCoef = np.empty((len(ncagx)+1,), dtype=float)
        newCoef[0] = 0.0
        newCoef[1:] = ncagx
        # Handle R_outer with
        # df/dx = df/d(x/R) * d(x/R)/dx = df/d(x/R) * 1/R
        newCoef /= self.R_outer
        return Zernike(newCoef, R_outer=self.R_outer, R_inner=self.R_inner)

    @lazy_property
    def gradY(self):
        """The y-derivative of this Zernike as a new Zernike object.
        """
        j = len(self.coef)-1
        ncagy = _noll_coef_array_grady(j, self.R_inner/self.R_outer).dot(self.coef[1:])
        newCoef = np.empty((len(ncagy)+1,), dtype=float)
        newCoef[0] = 0.0
        newCoef[1:] = ncagy
        # Handle R_outer with
        # df/dy = df/d(y/R) * d(y/R)/dy = df/d(y/R) * 1/R
        newCoef /= self.R_outer
        return Zernike(newCoef, R_outer=self.R_outer, R_inner=self.R_inner)

    def evalCartesian(self, x, y):
        """Evaluate this Zernike polynomial series at Cartesian coordinates x and y.

        @param x  x-coordinate of evaluation points.  Can be list-like.
        @param y  y-coordinate of evaluation points.  Can be list-like.
        @returns  Series evaluations as numpy array.
        """
        return horner2d(x, y, self._coef_array_xy, dtype=float)

    def evalPolar(self, rho, theta):
        """Evaluate this Zernike polynomial series at polar coordinates rho and theta.

        @param rho    radial coordinate of evaluation points.  Can be list-like.
        @param theta  azimuthal coordinate in radians (or as Angle object) of evaluation points.
                      Can be list-like.
        @returns  Series evaluations as numpy array.
        """
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return self.evalCartesian(x, y)

    def evalCartesianGrad(self, x, y):
        return self.gradX.evalCartesian(x, y), self.gradY.evalCartesian(x, y)

    def rotate(self, theta):
        """Return new Zernike polynomial series rotated by angle theta.  For example:

            >>> Z = Zernike(coefs)
            >>> Zrot = Z.rotate(theta)
            >>> Z.evalPolar(r, th) == Zrot.evalPolar(r, th + theta)

        @param theta  Angle in radians.
        @returns  A new Zernike object.
        """

        M = zernikeRotMatrix(len(self.coef)-1, theta)
        return Zernike(M.dot(self.coef), self.R_outer, self.R_inner)

    def __eq__(self, other):
        return (isinstance(other, Zernike) and
                np.array_equal(self.coef, other.coef) and
                self.R_outer == other.R_outer and
                self.R_inner == other.R_inner)

    def __hash__(self):
        return hash(("galsim.Zernike", tuple(self.coef), self.R_outer, self.R_inner))

    def __repr__(self):
        out = "galsim.zernike.Zernike("
        out += repr(self.coef)
        if self.R_outer != 1.0:
            out += ", R_outer={!r}".format(self.R_outer)
        if self.R_inner != 0.0:
            out += ", R_inner={!r}".format(self.R_inner)
        out += ")"
        return out


def zernikeRotMatrix(jmax, theta):
    """Construct Zernike basis rotation matrix.  This matrix can be used to convert a set of Zernike
    polynomial series coefficients expressed in one coordinate system to an equivalent set of
    coefficients expressed in a rotated coordinate system.  For example:

        >>> Z = Zernike(coefs)
        >>> R = zernikeRotMatrix(jmax, theta)
        >>> rotCoefs = R.dot(coefs)
        >>> Zrot = Zernike(rotCoefs)
        >>> Z.evalPolar(r, th) == Zrot.evalPolar(r, th + theta)

    Note that not all values of `jmax` are allowed.  For example, jmax=2 raises an Exception, since
    a non-zero Z_2 coefficient will in general rotate into a combination of Z_2 and Z_3
    coefficients, and therefore the needed rotation matrix requires jmax=3.  If you run into this
    kind of Exception, raising jmax by 1 will eliminate it.

    Also note that the returned matrix is intended to multiply a vector of Zernike coefficients
    `coef` indexed following the Noll (1976) convention, which starts at 1.  Since python sequences
    start at 0, we adopt the convention that `coef[0]` is unused, and `coef[i]` corresponds to the
    coefficient multiplying Z_i.  As a result, the size of the returned matrix is [jmax+1, jmax+1].

    @param jmax  Maximum Zernike index (in the Noll convention) over which to construct matrix.
    @param theta  Angle of rotation in radians.
    @returns  Matrix of size [jmax+1, jmax+1].
    """
    # Use formula from Tatulli (2013) arXiv:1302.7106v1

    # Note that coefficients mix if and only if they have the same radial index n and the same
    # absolute azimuthal index m.  This means that to construct a rotation matrix, we need for both
    # m's in a pair {(n, m), (n, -m)} to be represented, which places constraints on size.
    # Specifically, if the final Zernike indicated by size includes only one part of the pair, then
    # the rotation would mix coefficients into the element (size+1).  We simply disallow this here.

    n_jmax, m_jmax = noll_to_zern(jmax)
    # If m_jmax is zero, then we don't need to check the next element to ensure we have a complete
    # rotation matrix.
    if m_jmax != 0:
        n_jmaxp1, m_jmaxp1 = noll_to_zern(jmax+1)
        if n_jmax == n_jmaxp1 and abs(m_jmaxp1) == abs(m_jmax):
            raise GalSimValueError("Cannot construct Zernike rotation matrix for this jmax.", jmax)

    R = np.zeros((jmax+1, jmax+1), dtype=np.float64)
    R[0, 0] = 1.0
    for i in range(jmax):
        ni, mi = noll_to_zern(i+1)
        for j in range(max(0, i-1), min(i+2, jmax)):
            nj, mj = noll_to_zern(j+1)
            if ni != nj:
                continue
            if abs(mi) != abs(mj):
                continue
            if mi == mj:
                R[i+1, j+1] = np.cos(mj * theta)
            else:
                R[i+1, j+1] = np.sin(mj * theta)
    return R


def zernikeBasis(jmax, x, y, R_outer=1.0, R_inner=0.0):
    """Construct basis of Zernike polynomial series up to Noll index `jmax`, evaluated at a specific
    set of points `x` and `y`.  Useful for fitting Zernike polynomials to data, e.g.,

        >>> x, y, z = myDataToFit()
        >>> basis = zernikeBasis(11, x, y)
        >>> coefs, _, _, _ = np.linalg.lstsq(basis.T, z)
        >>> resids = Zernike(coefs).evalCartesian(x, y) - z

        or equivalently

        >>> resids = basis.T.dot(coefs).T - z

    Note that since we follow the Noll indexing scheme for Zernike polynomials, which begins at 1,
    but python sequences are indexed from 0, the length of the leading dimension in the result is
    `jmax+1` instead of `jmax`.  We somewhat arbitrarily fill the 0th slice along the first
    dimension with 0s (result[0, ...] == 0) so that it doesn't impact the results of np.linalg.lstsq
    as in the example above.

    @param  jmax     Maximum Noll index to use.
    @param  x        x-coordinates (can be list-like, congruent to y)
    @param  y        y-coordinates (can be list-like, congruent to x)
    @param  R_outer  Outer radius.  [default: 1.0]
    @param  R_inner  Inner radius.  [default: 0.0]
    @returns  [jmax+1, x.shape] array.  Slicing over first index gives basis vectors corresponding
              to individual Zernike polynomials.
    """
    R_outer = float(R_outer)
    R_inner = float(R_inner)
    eps = R_inner / R_outer

    noll_coef = _noll_coef_array_xy(jmax, eps)
    out = np.zeros(tuple((jmax+1,)+x.shape), dtype=float)
    out[1:] = np.array([horner2d(x/R_outer, y/R_outer, nc, dtype=float)
                        for nc in noll_coef.transpose(2,0,1)])
    return out
