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
from numbers import Real

from .utilities import LRU_Cache, binomial, horner2d, horner4d, nCr, lazy_property
from .integ import gq_annulus_points
from .errors import GalSimValueError, GalSimRangeError, GalSimIncompatibleValuesError

# Some utilities for working with Zernike polynomials

# Start off with the Zernikes up to j=15
_noll_n = [0,0,1,1,2,2,2,3,3,3,3,4,4,4,4,4]
_noll_m = [0,0,1,-1,0,-2,2,-1,1,-3,3,0,2,-2,4,-4]
def noll_to_zern(j):
    """Convert linear Noll index to tuple of Zernike indices.
    j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike
    index.

    c.f. https://oeis.org/A176988

    Parameters:
        j:      Zernike mode Noll index

    Returns:
        (n, m) tuple of Zernike indices
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

    Parameters:
        n:              Zernike radial coefficient
        m:              Zernike azimuthal coefficient
        obscuration:    Linear obscuration fraction.
        shape:          Output array shape

    Returns:
        2D array of coefficients in |r|^2 and r, where r = u + 1j * v, and u, v are unit
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

    Parameters:
        jmax:           Maximum Noll coefficient
        obscuration:    Linear obscuration fraction.

    Returns:
        2D array of coefficients in |r|^2 and r, where r = u + 1j * v, and u, v are unit
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

    Parameters:
        jmax:           Maximum Noll coefficient
        obscuration:    Linear obscuration fraction.

    Returns:
        2D array of coefficients in x and y.
    """
    maxn, _ = noll_to_zern(jmax)
    j_full = (maxn+1)*(maxn+2)//2
    if jmax < j_full:  # full row calculation may already be in cache
        return _noll_coef_array_xy(j_full, obscuration)[..., :jmax]

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

    Parameters:
        jmax:           Maximum Noll coefficient
        obscuration:    Linear obscuration fraction.

    Returns:
        2D array of coefficients in x and y.
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

    Parameters:
        jmax:           Maximum Noll coefficient
        obscuration:    Linear obscuration fraction.

    Returns:
        2D array of coefficients in x and y.
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
        return np.zeros((1,1), dtype=np.float64)
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
        return np.zeros((1,1), dtype=np.float64)
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


def describe_zernike(j):
    """Create a human-readable string describing the jth (unit circle) Zernike mode as a bivariate
    polynomial in x and y.

    Parameters:
        j:      Zernike mode Noll index

    Returns:
        string description of jth mode.
    """
    Z = Zernike([0]*j+[1])
    n, m = noll_to_zern(j)
    var = (1 if m==0 else 2)*(n+1)
    arr = Z._coef_array_xy/np.sqrt(var)
    first = True
    out = "sqrt({}) * (".format(var)

    for (i, k), val in np.ndenumerate(arr):
        if val != 0:
            if not first:
                out += " + "
            first = False
            ival = int(np.round(val))
            if ival != 1:
                out += str(int(np.round(val)))
            if i >= 1:
                out += "x"
            if i >= 2:
                out += "^"+str(i)
            if k >= 1:
                out += "y"
            if k >= 2:
                out += "^"+str(k)
    out += ")"
    # Clean up some special cases
    out = out.replace("(-1x", "(-x")
    out = out.replace("(-1y", "(-y")
    out = out.replace("+ -", "- ")
    if out == "sqrt(1) * ()":
        out = "sqrt(1) * (1)"
    return out


class Zernike:
    r"""A class to represent a Zernike polynomial series
    (http://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials).

    Zernike polynomials form an orthonormal basis over the unit circle.  The convention used here is
    for the normality constant to equal the area of integration, which is pi for the unit circle.
    I.e.,

    .. math::
        \int_\mathrm{unit circle} Z_i Z_j dA = \pi \delta_{i, j}.

    Two generalizations of the unit circle Zernike polynomials are also available in this class:
    annular Zernike polynomials, and polynomials defined over non-unit-radius circles.

    Annular Zernikes are orthonormal over an annulus instead of a circle (see Mahajan, J. Opt. Soc.
    Am. 71, 1, (1981)).  Similarly, the non-unit-radius polynomials are orthonormal over a region
    with outer radius not equal to 1.  Taken together, these generalizations yield the
    orthonormality condition:

    .. math::
        \int_\mathrm{annulus} Z_i Z_j dA =
        \pi \left(R_\mathrm{outer}^2 - R_\mathrm{inner}^2\right) \delta_{i, j}

    where :math:`0 <= R_\mathrm{inner} < R_\mathrm{outer}` indicate the inner and outer radii of
    the annulus over which the polynomials are orthonormal.

    The indexing convention for i and j above is that from Noll, J. Opt. Soc. Am. 66, 207-211(1976).
    Note that the Noll indices begin at 1; there is no Z_0.  Because of this, the series
    coefficients argument ``coef`` effectively begins with ``coef[1]`` (``coef[0]`` is ignored).
    This convention is used consistently throughout GalSim, e.g., `OpticalPSF`, `OpticalScreen`,
    `zernikeRotMatrix`, and `zernikeBasis`.

    As an example, the first few Zernike polynomials in terms of Cartesian coordinates x and y are

    ==========      ===========================
    Noll index            Polynomial
    ==========      ===========================
         1                  1
         2                  2 x
         3                  2 y
         4          sqrt(3) (2 (x^2 + y^2) - 1)
    ==========      ===========================

    A few mathematical convenience operations are additionally available.  Zernikes can be added,
    subtracted, or multiplied together, or multiplied by scalars.  Note, however, that two
    Zernikes can only be combined this way if they have matching ``R_outer`` and ``R_inner``
    attributes.  Zernike gradients, Laplacians and the determinant of the Hessian matrix are also
    available as properties that return new `Zernike` objects.

    Parameters:
        coef:       Zernike series coefficients.  Note that coef[i] corresponds to Z_i under the
                    Noll index convention, and coef[0] is ignored.  (I.e., coef[1] is 'piston',
                    coef[4] is 'defocus', ...)
        R_outer:    Outer radius.  [default: 1.0]
        R_inner:    Inner radius.  [default: 0.0]
    """
    def __init__(self, coef, R_outer=1.0, R_inner=0.0):
        self.coef = np.asarray(coef, dtype=float)
        if len(self.coef) <= 1:
            self.coef = np.array([0, 0], dtype=float)
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)

    @classmethod
    def _from_coef_array_xy(cls, coef_array_xy, R_outer=1.0, R_inner=0.0):
        """Construct a Zernike from a 2D array of Cartesian polynomial
        coefficients.
        """
        ret = Zernike.__new__(Zernike)
        ret._coef_array_xy = coef_array_xy
        ret.R_outer = R_outer
        ret.R_inner = R_inner
        return ret

    def __add__(self, rhs):
        """Add two Zernikes.

        Requires that each operand's ``R_outer`` and ``R_inner`` attributes are the same.
        """
        if not isinstance(rhs, Zernike):
            raise TypeError("Cannot add Zernike to type {}".format(type(rhs)))
        if self.R_outer != rhs.R_outer:
            raise ValueError("Cannot add Zernikes with inconsistent R_outer")
        if self.R_inner != rhs.R_inner:
            raise ValueError("Cannot add Zernikes with inconsistent R_inner")
        if 'coef' in self.__dict__:
            n = max(len(self.coef), len(rhs.coef))
            newCoef = np.zeros(n, dtype=float)
            newCoef[:len(self.coef)] = self.coef
            newCoef[:len(rhs.coef)] += rhs.coef
            return Zernike(newCoef, R_outer=self.R_outer, R_inner=self.R_inner)
        else:
            s1 = self._coef_array_xy.shape
            s2 = rhs._coef_array_xy.shape
            sout = max(s1[0], s2[0]), max(s1[1], s2[1])
            coef_array_xy = np.zeros(sout, dtype=float)
            coef_array_xy[:s1[0], :s1[1]] = self._coef_array_xy
            coef_array_xy[:s2[0], :s2[1]] += rhs._coef_array_xy
            return Zernike._from_coef_array_xy(
                coef_array_xy,
                R_outer=self.R_outer,
                R_inner=self.R_inner
            )

    def __sub__(self, rhs):
        """Subtract two Zernikes.

        Requires that each operand's ``R_outer`` and ``R_inner`` attributes are the same.
        """
        return self + (-rhs)

    def __neg__(self):
        """Negate a Zernike.
        """
        if 'coef' in self.__dict__:
            ret = Zernike(-self.coef, R_outer=self.R_outer, R_inner=self.R_inner)
            if '_coef_array_xy' in self.__dict__:
                ret._coef_array_xy = -self._coef_array_xy
        else:
            ret = Zernike._from_coef_array_xy(
                -self._coef_array_xy,
                R_outer=self.R_outer,
                R_inner=self.R_inner
            )
        return ret

    def __mul__(self, rhs):
        """Multiply two Zernikes, or multiply a Zernike by a scalar.

        If both operands are Zernikes, then the ``R_outer`` and ``R_inner`` attributes of each must
        be the same.
        """
        if isinstance(rhs, Real):
            if 'coef' in self.__dict__:
                ret = Zernike(rhs*self.coef, self.R_outer, self.R_inner)
                if '_coef_array_xy' in self.__dict__:
                    ret._coef_array_xy = rhs*self._coef_array_xy
            else:
                ret = Zernike._from_coef_array_xy(
                    rhs*self._coef_array_xy,
                    R_outer=self.R_outer,
                    R_inner=self.R_inner
                )
            return ret
        elif isinstance(rhs, Zernike):
            if self.R_outer != rhs.R_outer:
                raise ValueError("Cannot multiply Zernikes with inconsistent R_outer")
            if self.R_inner != rhs.R_inner:
                raise ValueError("Cannot multiply Zernikes with inconsistent R_inner")
            n1, _ = noll_to_zern(len(self.coef)-1)
            n2, _ = noll_to_zern(len(rhs.coef)-1)
            nTarget = n1+n2  # Maximum possible radial degree is sum of input radial degrees
            jTarget = (nTarget+1)*(nTarget+2)//2  # Largest Noll index with above radial degree

            # To multiply, we first convolve in 2D the xy coefficients of each polynomial
            sxy = self._coef_array_xy
            rxy = rhs._coef_array_xy
            shape = (sxy.shape[0]+rxy.shape[0]-1,
                     sxy.shape[1]+rxy.shape[1]-1)
            newXY = np.zeros(shape, dtype=float)
            if sxy.shape[0]*sxy.shape[1] > rxy.shape[0]*rxy.shape[1]:
                sxy, rxy = rxy, sxy
            for (i, j), c in np.ndenumerate(sxy):
                newXY[i:i+rxy.shape[0], j:j+rxy.shape[1]] += c*rxy

            return Zernike._from_coef_array_xy(
                newXY,
                R_outer=self.R_outer,
                R_inner=self.R_inner
            )
        else:
            raise TypeError("Cannot multiply Zernike by type {}".format(type(rhs)))

    def __rmul__(self, rhs):
        """Equivalent to obj * rhs.  See `__mul__` for details."""
        return self*rhs

    def __truediv__(self, rhs):
        if not isinstance(rhs, Real):
            raise TypeError("Cannot multiply Zernike by type {}".format(type(rhs)))
        return self*(1./rhs)

    @lazy_property
    def coef(self):
        """Zernike series coefficients.

        Note that coef[i] corresponds to Z_i under the Noll index convention, and coef[0] is
        ignored.  (I.e., coef[1] is 'piston', coef[4] is 'defocus', ...).
        """

        # The strategy is to use the orthonormality of the Zernike polynomials and
        # integrate over the annulus.  In particular,
        # int_annulus xy(x,y) Zernike_j(x,y) dx dy = area_annulus a_j
        # defines coefficients a_j in the expansion
        # xy(x, y) = \sum_j a_j Zernike_j(x,y).

        # We can use Gaussian Quadrature over an annulus to do the integration.

        # First determine the number of quadrature points needed to integrate up to
        # the maximum possible radial degree.
        xy = self._coef_array_xy
        nTarget = max(*xy.shape)-1  # Maximum radial degree
        jTarget = (nTarget+1)*(nTarget+2)//2  # Largest Noll index with above radial degree

        nRings = nTarget//2+1
        nSpokes = 2*nTarget+1
        x, y, weights = gq_annulus_points(self.R_outer, self.R_inner, nRings, nSpokes)
        val = horner2d(x, y, xy, dtype=float)
        basis = zernikeBasis(
            jTarget, x, y, R_outer=self.R_outer, R_inner=self.R_inner
        )
        area = np.pi*(self.R_outer**2 - self.R_inner**2)
        return np.dot(basis, val*weights/area)

    @lazy_property
    def hessian(self):
        """The determinant of the Hessian matrix of this Zernike polynomial expressed as a new
        Zernike polynomial.  The Hessian matrix is the matrix of second derivatives, to the
        determinant is d^2Z/dx^2 * d^Z/dy^2 - (d^Z/dxdy)^2, and is an expression of the local
        curvature of the Zernike polynomial.
        """
        dxx = self.gradX.gradX
        dxy = self.gradX.gradY
        dyy = self.gradY.gradY
        return dxx*dyy - dxy*dxy

    @lazy_property
    def laplacian(self):
        """The Laplacian of this Zernike polynomial expressed as a new Zernike polynomial. The
        Laplacian is d^2Z/dx^2 + d^2Z/dy^2 (the trace of the Hessian matrix), and is an expression
        of the local divergence of the Zernike polynomial.
        """
        return self.gradX.gradX + self.gradY.gradY

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

    def __call__(self, x, y):
        """Evaluate this Zernike polynomial series at Cartesian coordinates x and y.
        Synonym for `evalCartesian`.

        Parameters:
            x:    x-coordinate of evaluation points.  Can be list-like.
            y:    y-coordinate of evaluation points.  Can be list-like.
        Returns:
            Series evaluations as numpy array.
        """
        return self.evalCartesian(x, y)

    def evalCartesian(self, x, y):
        """Evaluate this Zernike polynomial series at Cartesian coordinates x and y.

        Parameters:
            x:    x-coordinate of evaluation points.  Can be list-like.
            y:    y-coordinate of evaluation points.  Can be list-like.

        Returns:
            Series evaluations as numpy array.
        """
        return horner2d(x, y, self._coef_array_xy, dtype=float)

    def evalPolar(self, rho, theta):
        """Evaluate this Zernike polynomial series at polar coordinates rho and theta.

        Parameters:
            rho:      radial coordinate of evaluation points.  Can be list-like.
            theta:    azimuthal coordinate in radians (or as `Angle` object) of evaluation points.
                      Can be list-like.

        Returns:
            Series evaluations as numpy array.
        """
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return self.evalCartesian(x, y)

    def evalCartesianGrad(self, x, y):
        """Evaluate the gradient of this Zernike polynomial series at cartesian coordinates
        x and y.

        Parameters:
            x:  x-coordinate of evaluation points.  Can be list-like.
            y:  y-coordinate of evaluation points.  Can be list-like.
        Returns:
            Tuple of arrays for x-gradient and y-gradient.
        """
        return self.gradX.evalCartesian(x, y), self.gradY.evalCartesian(x, y)

    def rotate(self, theta):
        """Return new Zernike polynomial series rotated by angle theta.

        For example::

            >>> Z = Zernike(coefs)
            >>> Zrot = Z.rotate(theta)
            >>> Z.evalPolar(r, th) == Zrot.evalPolar(r, th + theta)

        Parameters:
            theta:    angle in radians.

        Returns:
            A new Zernike object.
        """
        M = zernikeRotMatrix(len(self.coef)-1, theta)
        return Zernike(M.dot(self.coef), self.R_outer, self.R_inner)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, Zernike) and
                 np.array_equal(self.coef, other.coef) and
                 self.R_outer == other.R_outer and
                 self.R_inner == other.R_inner))

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
    coefficients expressed in a rotated coordinate system.

    For example::

        >>> Z = Zernike(coefs)
        >>> R = zernikeRotMatrix(jmax, theta)
        >>> rotCoefs = R.dot(coefs)
        >>> Zrot = Zernike(rotCoefs)
        >>> Z.evalPolar(r, th) == Zrot.evalPolar(r, th + theta)

    Note that not all values of ``jmax`` are allowed.  For example, jmax=2 raises an Exception,
    since a non-zero Z_2 coefficient will in general rotate into a combination of Z_2 and Z_3
    coefficients, and therefore the needed rotation matrix requires jmax=3.  If you run into this
    kind of Exception, raising jmax by 1 will eliminate it.

    Also note that the returned matrix is intended to multiply a vector of Zernike coefficients
    ``coef`` indexed following the Noll (1976) convention, which starts at 1.  Since python
    sequences start at 0, we adopt the convention that ``coef[0]`` is unused, and ``coef[i]``
    corresponds to the coefficient multiplying Z_i.  As a result, the size of the returned matrix
    is [jmax+1, jmax+1].

    Parameters:
        jmax:   Maximum Zernike index (in the Noll convention) over which to construct matrix.
        theta:  angle of rotation in radians.

    Returns:
        Matrix of size [jmax+1, jmax+1].
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
    """Construct basis of Zernike polynomial series up to Noll index ``jmax``, evaluated at a
    specific set of points ``x`` and ``y``.

    Useful for fitting Zernike polynomials to data, e.g.::

        >>> x, y, z = myDataToFit()
        >>> basis = zernikeBasis(11, x, y)
        >>> coefs, _, _, _ = np.linalg.lstsq(basis.T, z)
        >>> resids = Zernike(coefs).evalCartesian(x, y) - z

    or equivalently::

        >>> resids = basis.T.dot(coefs).T - z

    Note that since we follow the Noll indexing scheme for Zernike polynomials, which begins at 1,
    but python sequences are indexed from 0, the length of the leading dimension in the result is
    ``jmax+1`` instead of ``jmax``.  We somewhat arbitrarily fill the 0th slice along the first
    dimension with 0s (result[0, ...] == 0) so that it doesn't impact the results of
    ``np.linalg.lstsq`` as in the example above.

    Parameters:
         jmax:      Maximum Noll index to use.
         x:         x-coordinates (can be list-like, congruent to y)
         y:         y-coordinates (can be list-like, congruent to x)
         R_outer:   Outer radius.  [default: 1.0]
         R_inner:   Inner radius.  [default: 0.0]

    Returns:
        [jmax+1, x.shape] array.  Slicing over first index gives basis vectors corresponding
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


def zernikeGradBases(jmax, x, y, R_outer=1.0, R_inner=0.0):
    """Construct bases of Zernike polynomial series gradients up to Noll index ``jmax``, evaluated
    at a specific set of points ``x`` and ``y``.

    Note that since we follow the Noll indexing scheme for Zernike polynomials, which begins at 1,
    but python sequences are indexed from 0, the length of the leading dimension in the result is
    ``jmax+1`` instead of ``jmax``.  We somewhat arbitrarily fill the 0th slice along the first
    dimension with 0s (result[0, ...] == 0).

    Parameters:
         jmax:      Maximum Noll index to use.
         x:         x-coordinates (can be list-like, congruent to y)
         y:         y-coordinates (can be list-like, congruent to x)
         R_outer:   Outer radius.  [default: 1.0]
         R_inner:   Inner radius.  [default: 0.0]

    Returns:
        [2, jmax+1, x.shape] array.  The first index selects the gradient in the x/y direction,
        slicing over the second index gives basis vectors corresponding to individual Zernike
        polynomials.
    """

    R_outer = float(R_outer)
    R_inner = float(R_inner)
    eps = R_inner / R_outer

    noll_coef_x = _noll_coef_array_xy_gradx(jmax, eps)
    dzkdx = np.zeros(tuple((jmax + 1,) + x.shape), dtype=float)
    dzkdx[1:] = np.array([
        horner2d(x/R_outer, y/R_outer, nc, dtype=float)/R_outer
        for nc in noll_coef_x.transpose(2, 0, 1)
    ])

    noll_coef_y = _noll_coef_array_xy_grady(jmax, eps)
    dzkdy = np.zeros(tuple((jmax + 1,) + x.shape), dtype=float)
    dzkdy[1:] = np.array([
        horner2d(x/R_outer, y/R_outer, nc, dtype=float)/R_outer
        for nc in noll_coef_y.transpose(2, 0, 1)
    ])
    return np.array([dzkdx, dzkdy])


class DoubleZernike:
    r"""The Cartesian product of two (annular) Zernike polynomial series.  Each
    double Zernike term is the product of two single Zernike polynomials over
    separate annuli:

    .. math::
        DZ_{k,j}(u, v, x, y) = Z_k(u, v) Z_j(x, y)

    The double Zernike's therefore satisfy the orthonormality condition:

    .. math::
        \int_{annuli} DZ_{k,j} DZ_{k',j'} = A_1 A_2 \delta_{k, k'} \delta_{j, j'}

    where :math:`A_1` and :math:`A_2` are the areas of the two annuli.

    Double Zernikes series are useful for representing the field and pupil
    dependent wavefronts of a telescope.  We adopt the typical convention that
    the first index (the ``k`` index) corresponds to the field-dependence, while
    the second index (the ``j`` index) corresponds to the pupil-dependence.

    Parameters:
        coef:       [kmax, jmax] array of double Zernike coefficients.  Like the
                    single Zernike class, the 0th index of either axis is
                    ignored.
        uv_outer:   Outer radius of the first annulus.  [default: 1.0]
        uv_inner:   Inner radius of the first annulus.  [default: 0.0]
        xy_outer:   Outer radius of the second annulus.  [default: 1.0]
        xy_inner:   Inner radius of the second annulus.  [default: 0.0]
    """
    def __init__(
        self,
        coef,
        *,
        uv_outer=1.0,  # "field"
        uv_inner=0.0,
        xy_outer=1.0,  # "pupil"
        xy_inner=0.0
    ):
        self.coef = np.asarray(coef, dtype=float)
        self.uv_outer = uv_outer
        self.uv_inner = uv_inner
        self.xy_outer = xy_outer
        self.xy_inner = xy_inner
        self._xy_series = [
            Zernike(col, R_outer=uv_outer, R_inner=uv_inner) for col in coef.T
        ]

    @lazy_property
    def _kmax(self):
        """Maximum Noll index of the field dependence."""
        if 'coef' in self.__dict__:
            return self.coef.shape[0] - 1
        else:
            sh = self._coef_array_uvxy.shape
            nuv = max(sh[0], sh[1]) - 1  # Max radial degree of uv
            return (nuv+1)*(nuv+2)//2

    @lazy_property
    def _jmax(self):
        """Maximum Noll index of the pupil dependence."""
        if 'coef' in self.__dict__:
            return self.coef.shape[1] - 1
        else:
            sh = self._coef_array_uvxy.shape
            nxy = max(sh[2], sh[3]) - 1  # Max radial degree of xy
            return (nxy+1)*(nxy+2)//2

    @lazy_property
    def _nuv(self):
        """Maximum radial degree of the field dependence."""
        return noll_to_zern(self._kmax)[0]

    @lazy_property
    def _nxy(self):
        """Maximum radial degree of the pupil dependence."""
        return noll_to_zern(self._jmax)[0]

    @classmethod
    def _from_uvxy(
        cls,
        uvxy,
        *,
        uv_outer=1.0,  # "field"
        uv_inner=0.0,
        xy_outer=1.0,  # "pupil"
        xy_inner=0.0
    ):
        """Construct a DoubleZernike from a 4D array of Cartesian polynomial
        coefficients.
        """
        ret = DoubleZernike.__new__(DoubleZernike)
        ret._coef_array_uvxy = np.asarray(uvxy, dtype=float)
        ret.uv_outer = uv_outer
        ret.uv_inner = uv_inner
        ret.xy_outer = xy_outer
        ret.xy_inner = xy_inner
        return ret

    def _call_old(self, u, v, x=None, y=None):
        # Original implementation constructing "single" Zernike from
        # coefficients directly.  Retained mostly for testing purposes.
        assert np.ndim(u) == np.ndim(v)
        assert np.shape(u) == np.shape(v)
        if x is None:  # uv only
            if np.ndim(u) == 0:  # uv scalar
                return Zernike(
                    [z(u, v) for z in self._xy_series],
                    R_outer=self.xy_outer,
                    R_inner=self.xy_inner
                )
            else:  # uv vector
                return [
                    Zernike(
                        [z(u[i], v[i]) for z in self._xy_series],
                        R_outer=self.xy_outer,
                        R_inner=self.xy_inner
                    ) for i in range(len(u))
                ]
        else:  # uv and xy
            assert np.ndim(x) == np.ndim(y)
            assert np.shape(x) == np.shape(y)
            if np.ndim(u) == 0:  # uv scalar
                return self._call_old(u, v)(x, y)  # defer to Zernike.__call__
            else:  # uv vector
                # Note that we _don't_ defer to Zernike.__call__, as doing so
                # would yield the outer product of uv and xy, which is not what
                # we want.
                zs = self._call_old(u, v)
                if np.ndim(x) == 0:  # xy scalar
                    return np.array([z(x, y) for z in zs])
                else:  # xy vector
                    assert np.shape(x) == np.shape(u)
                    return np.array([z(x[i], y[i]) for i, z in enumerate(zs)])

    def _compute_coef(self, kmax, jmax):
        # Same strategy as Zernike; take advantage of orthonormality to
        # determine Noll coefficients from Cartesian coefficients.

        # Determine number of GQ points
        uv_rings = self._nuv//2+1
        uv_spokes = 2*self._nuv+1
        xy_rings = self._nxy//2+1
        xy_spokes = 2*self._nxy+1

        # Compute GQ points and weights on double annulus
        u, v, uv_w = gq_annulus_points(self.uv_outer, self.uv_inner, uv_rings, uv_spokes)
        x, y, xy_w = gq_annulus_points(self.xy_outer, self.xy_inner, xy_rings, xy_spokes)
        nu = len(u)
        nx = len(x)
        u = np.repeat(u, nx)
        v = np.repeat(v, nx)
        uv_w = np.repeat(uv_w, nx)
        x = np.tile(x, nu)
        y = np.tile(y, nu)
        xy_w = np.tile(xy_w, nu)
        weights = uv_w * xy_w

        # Evaluate uvxy polynomial at GQ points
        vals = horner4d(u, v, x, y, self._coef_array_uvxy)

        # Project into Zernike basis
        basis = doubleZernikeBasis(
            kmax, jmax,
            u, v, x, y,
            uv_outer=self.uv_outer, uv_inner=self.uv_inner,
            xy_outer=self.xy_outer, xy_inner=self.xy_inner
        )
        area = np.pi**2 * (self.uv_outer**2 - self.uv_inner**2) * (self.xy_outer**2 - self.xy_inner**2)
        return np.dot(basis, vals*weights/area)


    @lazy_property
    def coef(self):
        """DoubleZernike coefficients.

        The coefficients are stored in a 2D array, where the first index
        corresponds to the ``uv`` dependence and the second index corresponds to
        the ``xy`` dependence.  The indices are Noll indices.  As an example, when
        describing a telescope wavefront the (1, 4) tuple corresponds to a
        constant (over the field) pupil defocus.  The (2, 5) term is a linear
        (over the field) astigmatism, etc.
        """
        return self._compute_coef(self._kmax, self._jmax)

    @lazy_property
    def _coef_array_uvxy(self):
        uv_shape = self._xy_series[0]._coef_array_xy.shape
        xy_shape = Zernike([0]*self._jmax+[1])._coef_array_xy.shape
        out_shape = uv_shape + xy_shape
        out = np.zeros(out_shape, dtype=float)
        # Now accumulate one uv term at a time
        for j, zk in enumerate(self._xy_series):
            coef_array_uv = zk._coef_array_xy
            coef_array_xy = Zernike(
                [0]*j+[1],
                R_outer=self.xy_outer,
                R_inner=self.xy_inner,
            )._coef_array_xy
            term = np.multiply.outer(coef_array_uv, coef_array_xy)
            sh = term.shape
            out[:sh[0], :sh[1], :sh[2], :sh[3]] += term
        return out

    def __call__(self, u, v, x=None, y=None):
        """Evaluate this DoubleZernike polynomial series at Cartesian
        coordinates u, v, x, y.

        Parameters:
            u, v: float or array-like.  Coordinates on first annulus.
            x, y: float or array-like.  Coordinates on second annulus.

        Returns:
            float or array-like.  Value of the DoubleZernike polynomial series.
        """
        # Cases:
        # uv only:
        #  if uv scalar, then return Zernike
        #  if uv vector, then return list of Zernike
        # uv and xy:
        #  if uv scalar:
        #    if xy scalar, then return scalar
        #    if xy vector, then return vector
        #  if uv vector:
        #   if xy scalar, then return vector
        #   if xy vector, then return vector, uv and xy must be congruent
        assert np.ndim(u) == np.ndim(v)
        assert np.shape(u) == np.shape(v)
        if (x is None) != (y is None):
            raise GalSimIncompatibleValuesError(
                "Must provide both x and y, or neither.",
                x=x, y=y
            )
        if x is None:
            if np.ndim(u) == 0:
                a_ij = np.zeros(self._coef_array_uvxy.shape[2:4])
                for i, j in np.ndindex(a_ij.shape):
                    a_ij[i, j] = horner2d(
                        u, v, self._coef_array_uvxy[..., i, j], dtype=float
                    )
                return Zernike._from_coef_array_xy(
                    a_ij,
                    R_outer=self.xy_outer,
                    R_inner=self.xy_inner
                )
            else:
                return [
                    self.__call__(u_, v_) for u_, v_ in zip(u, v)
                ]
        else:
            assert np.ndim(x) == np.ndim(y)
            assert np.shape(x) == np.shape(y)
            if np.ndim(u) == 0:  # uv scalar
                return self.__call__(u, v)(x, y)
            else:  # uv vector
                if np.ndim(x) == 0:  # xy scalar
                    return np.array([z(x, y) for z in self.__call__(u, v)])
                else: # xy vector
                    assert np.shape(x) == np.shape(u)
                    return horner4d(u, v, x, y, self._coef_array_uvxy)

    def __neg__(self):
        """Negate a DoubleZernike."""
        if 'coef' in self.__dict__:
            ret = DoubleZernike(
                -self.coef,
                uv_outer=self.uv_outer, uv_inner=self.uv_inner,
                xy_outer=self.xy_outer, xy_inner=self.xy_inner
            )
            if '_coef_array_uvxy' in self.__dict__:
                ret._coef_array_uvxy = -self._coef_array_uvxy
        else:
            ret = DoubleZernike._from_uvxy(
                -self._coef_array_uvxy,
                uv_outer=self.uv_outer, uv_inner=self.uv_inner,
                xy_outer=self.xy_outer, xy_inner=self.xy_inner
            )
        return ret

    def __add__(self, rhs):
        """Add two DoubleZernikes.

        Requires that each operand's xy and uv domains are the same.
        """
        if not isinstance(rhs, DoubleZernike):
            raise TypeError("Cannot add DoubleZernike to type {}".format(type(rhs)))
        if self.uv_outer != rhs.uv_outer:
            raise ValueError("Cannot add DoubleZernikes with inconsistent uv_outer")
        if self.uv_inner != rhs.uv_inner:
            raise ValueError("Cannot add DoubleZernikes with inconsistent uv_inner")
        if self.xy_outer != rhs.xy_outer:
            raise ValueError("Cannot add DoubleZernikes with inconsistent xy_outer")
        if self.xy_inner != rhs.xy_inner:
            raise ValueError("Cannot add DoubleZernikes with inconsistent xy_inner")
        if 'coef' in self.__dict__:
            kmax = max(self._kmax, rhs._kmax)
            jmax = max(self._jmax, rhs._jmax)
            newCoef = np.zeros((kmax+1, jmax+1), dtype=float)
            newCoef[:self._kmax+1, :self._jmax+1] = self.coef
            newCoef[:rhs._kmax+1, :rhs._jmax+1] += rhs.coef
            return DoubleZernike(
                newCoef,
                uv_outer=self.uv_outer, uv_inner=self.uv_inner,
                xy_outer=self.xy_outer, xy_inner=self.xy_inner
            )
        else:
            s1 = self._coef_array_uvxy.shape
            s2 = rhs._coef_array_uvxy.shape
            sh = [max(s1[i], s2[i]) for i in range(4)]
            coef_array_uvxy = np.zeros(sh, dtype=float)
            coef_array_uvxy[:s1[0], :s1[1], :s1[2], :s1[3]] = self._coef_array_uvxy
            coef_array_uvxy[:s2[0], :s2[1], :s2[2], :s2[3]] += rhs._coef_array_uvxy
            return DoubleZernike._from_uvxy(
                coef_array_uvxy,
                uv_outer=self.uv_outer, uv_inner=self.uv_inner,
                xy_outer=self.xy_outer, xy_inner=self.xy_inner
            )

    def __sub__(self, rhs):
        """Subtract two DoubleZernikes.

        Requires that each operand's xy and uv domains are the same.
        """
        return self + (-rhs)

    def __mul__(self, rhs):
        """Multiply two DoubleZernikes, or multiply a DoubleZernike by a scalar.

        If both operands are DoubleZernikes, then the domains for both annuli
        must be the same.
        """
        if isinstance(rhs, Real):
            if 'coef' in self.__dict__:
                ret = DoubleZernike(
                    rhs*self.coef,
                    uv_outer=self.uv_outer, uv_inner=self.uv_inner,
                    xy_outer=self.xy_outer, xy_inner=self.xy_inner
                )
                if '_coef_array_uvxy' in self.__dict__:
                    ret._coef_array_uvxy = rhs*self._coef_array_uvxy
            else:
                ret = DoubleZernike._from_uvxy(
                    rhs*self._coef_array_uvxy,
                    uv_outer=self.uv_outer, uv_inner=self.uv_inner,
                    xy_outer=self.xy_outer, xy_inner=self.xy_inner
                )
            return ret
        elif isinstance(rhs, DoubleZernike):
            if self.uv_outer != rhs.uv_outer:
                raise ValueError("Cannot multiply DoubleZernikes with inconsistent uv_outer")
            if self.uv_inner != rhs.uv_inner:
                raise ValueError("Cannot multiply DoubleZernikes with inconsistent uv_inner")
            if self.xy_outer != rhs.xy_outer:
                raise ValueError("Cannot multiply DoubleZernikes with inconsistent xy_outer")
            if self.xy_inner != rhs.xy_inner:
                raise ValueError("Cannot multiply DoubleZernikes with inconsistent xy_inner")
            # Multiplication is a 4d convolution of the Cartesian coefficients.
            # Easiest to get it right just by hand...
            uvxy1 = self._coef_array_uvxy
            uvxy2 = rhs._coef_array_uvxy
            sh1 = uvxy1.shape
            sh2 = uvxy2.shape
            outshape = tuple([d0+d1-1 for d0, d1 in zip(sh1, sh2)])
            uvxy = np.zeros(outshape)
            for (i, j, k, l), c in np.ndenumerate(uvxy1):
                uvxy[i:i+sh2[0], j:j+sh2[1], k:k+sh2[2], l:l+sh2[3]] += c*uvxy2
            return DoubleZernike._from_uvxy(
                uvxy,
                uv_outer=self.uv_outer, uv_inner=self.uv_inner,
                xy_outer=self.xy_outer, xy_inner=self.xy_inner
            )
        else:
            raise TypeError("Cannot multiply DoubleZernike by type {}".format(type(rhs)))

    def __rmul__(self, rhs):
        """Equivalent to obj * rhs.  See `__mul__` for details."""
        return self*rhs

    def __truediv__(self, rhs):
        if not isinstance(rhs, Real):
            raise TypeError("Cannot multiply Zernike by type {}".format(type(rhs)))
        return self*(1./rhs)

    def __eq__(self, rhs):
        if not isinstance(rhs, DoubleZernike):
            return False
        return (
            np.array_equal(self.coef, rhs.coef) and
            self.uv_outer == rhs.uv_outer and
            self.uv_inner == rhs.uv_inner and
            self.xy_outer == rhs.xy_outer and
            self.xy_inner == rhs.xy_inner
        )

    def __hash__(self):
        return hash((
            "galsim.DoubleZernike",
            tuple(self.coef.ravel()),
            self.coef.shape,
            self.uv_outer,
            self.uv_inner,
            self.xy_outer,
            self.xy_inner
        ))

    def __repr__(self):
        out = "galsim.zernike.DoubleZernike("
        out += repr(self.coef)
        if self.uv_outer != 1.0:
            out += ", uv_outer={}".format(self.uv_outer)
        if self.uv_inner != 0.0:
            out += ", uv_inner={}".format(self.uv_inner)
        if self.xy_outer != 1.0:
            out += ", xy_outer={}".format(self.xy_outer)
        if self.xy_inner != 0.0:
            out += ", xy_inner={}".format(self.xy_inner)
        out += ")"
        return out

    @lazy_property
    def gradX(self):
        """The gradient of the DoubleZernike in the x direction."""
        uvxy = self._coef_array_uvxy
        gradx = np.zeros_like(uvxy)
        for (i, j, k, l), c in np.ndenumerate(uvxy):
            if k > 0:
                if c != 0:
                    gradx[i, j, k-1, l] = c*k
        return DoubleZernike._from_uvxy(
            gradx,
            uv_outer=self.uv_outer, uv_inner=self.uv_inner,
            xy_outer=self.xy_outer, xy_inner=self.xy_inner
        )

    @lazy_property
    def gradY(self):
        """The gradient of the DoubleZernike in the y direction."""
        uvxy = self._coef_array_uvxy
        grady = np.zeros_like(uvxy)
        for (i, j, k, l), c in np.ndenumerate(uvxy):
            if l > 0:
                if c != 0:
                    grady[i, j, k, l-1] = c*l
        return DoubleZernike._from_uvxy(
            grady,
            xy_outer=self.xy_outer, xy_inner=self.xy_inner,
            uv_outer=self.uv_outer, uv_inner=self.uv_inner
        )

    @lazy_property
    def gradU(self):
        """The gradient of the DoubleZernike in the u direction."""
        uvxy = self._coef_array_uvxy
        gradu = np.zeros_like(uvxy)
        for (i, j, k, l), c in np.ndenumerate(uvxy):
            if i > 0:
                if c != 0:
                    gradu[i-1, j, k, l] = c*i
        return DoubleZernike._from_uvxy(
            gradu,
            xy_outer=self.xy_outer, xy_inner=self.xy_inner,
            uv_outer=self.uv_outer, uv_inner=self.uv_inner
        )

    @lazy_property
    def gradV(self):
        """The gradient of the DoubleZernike in the v direction."""
        uvxy = self._coef_array_uvxy
        gradv = np.zeros_like(uvxy)
        for (i, j, k, l), c in np.ndenumerate(uvxy):
            if j > 0:
                if c != 0:
                    gradv[i, j-1, k, l] = c*j
        return DoubleZernike._from_uvxy(
            gradv,
            xy_outer=self.xy_outer, xy_inner=self.xy_inner,
            uv_outer=self.uv_outer, uv_inner=self.uv_inner
        )

    @lazy_property
    def mean_xy(self):
        """Mean value over the xy annulus, returned as a Zernike in the uv
        domain."""

        # For any Zernike series, the expectation value is just the coefficient
        # of the Z1 term.  All the other terms have zero expectation.  For the
        # double Zernike series, the uv dependence of the xy expectation
        # value is contained in the first column of the coefficient array.

        if 'coef' in self.__dict__:
            c = self.coef[:, 1]
        else:
            c = self._compute_coef(self._kmax, 1)[:, 1]
        return Zernike(c, R_outer=self.uv_outer, R_inner=self.uv_inner)

    @lazy_property
    def mean_uv(self):
        """Mean value over the uv annulus, returned as a Zernike in the xy
        domain."""

        # See comment in mean_xy.

        if 'coef' in self.__dict__:
            c = self.coef[1]
        else:
            c = self._compute_coef(1, self._jmax)[1]
        return Zernike(c, R_outer=self.xy_outer, R_inner=self.xy_inner)

    def rotate(self, *, theta_uv=0.0, theta_xy=0.0):
        """Rotate the DoubleZernike by the given angle(s).

        For example:

            >>> DZrot = DZ.rotate(theta_xy=th)
            >>> DZ(u, v, r*np.cos(ph), r*np.sin(ph)) == DZrot(u, v, r*np.cos(ph+th), r*np.sin(ph+th))

        or:

            >>> DZrot = DZ.rotate(theta_uv=th)
            >>> DZ(r*np.cos(ph), r*np.sin(ph), x, y) == DZrot(r*np.cos(ph+th), r*np.sin(ph+th), x, y)

        Parameters:
            theta_uv:  Rotation angle (in radians) in the uv plane.
            theta_xy:  Rotation angle (in radians) in the xy plane.

        Returns:
            The rotated DoubleZernike.
        """
        M_uv = zernikeRotMatrix(self._kmax, theta_uv)
        M_xy = zernikeRotMatrix(self._jmax, theta_xy)
        coef = M_uv @ self.coef @ M_xy.T
        return DoubleZernike(
            coef,
            uv_outer=self.uv_outer, uv_inner=self.uv_inner,
            xy_outer=self.xy_outer, xy_inner=self.xy_inner
        )


def doubleZernikeBasis(
    kmax, jmax, u, v, x, y, *, uv_outer=1.0, uv_inner=0.0, xy_outer=1.0, xy_inner=0.0
):
    """Construct basis of DoubleZernike polynomial series up to Noll indices
    (kmax, jmax), evaluated at (u, v, x, y).

    Parameters:
        kmax:  Maximum Noll index for first annulus.
        jmax:  Maximum Noll index for second annulus.
        u, v: Coordinates in the first annulus.
        x, y: Coordinates in the second annulus.
        uv_outer:  Outer radius of the first annulus.
        uv_inner:  Inner radius of the first annulus.
        xy_outer:  Outer radius of the second annulus.
        xy_inner:  Inner radius of the second annulus.

    Returns:
        [kmax+1, jmax+1, u.shape] array.  Slicing over the first two dimensions
        gives the basis vectors corresponding to individual DoubleZernike terms.

    """
    return np.einsum(
        'k...,j...->kj...',
        zernikeBasis(kmax, u, v, R_outer=uv_outer, R_inner=uv_inner),
        zernikeBasis(jmax, x, y, R_outer=xy_outer, R_inner=xy_inner)
    )
