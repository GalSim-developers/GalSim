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
import functools
from contextlib import contextmanager
import weakref
import os
import warnings
import numpy as np
import pstats
import math
import heapq
import multiprocessing
import cProfile, pstats
from io import StringIO
import logging
import time
from collections.abc import Hashable
from collections import Counter
import pickle
import copy

from . import _galsim
from .errors import GalSimError, GalSimValueError, GalSimIncompatibleValuesError, GalSimRangeError
from .errors import galsim_warn
from .position import Position, PositionD, _PositionD
from .angle import AngleUnit, arcsec
from .image import Image
from .table import trapz, _LookupTable, LookupTable2D
from .wcs import JacobianWCS, PixelScale
from .random import BaseDeviate, UniformDeviate
from . import meta_data

# A couple things are documented as galsim.utilties.* functions, but live in other files.
# Bring them into scope here.
from .interpolant import convert_interpolant
from .table import find_out_of_bounds_position
from .position import parse_pos_args
from ._utilities import *


def roll2d(image, shape):
    """Perform a 2D roll (circular shift) on a supplied 2D NumPy array, conveniently.

    Parameters:
        image:      The NumPy array to be circular shifted.
        shape:      (iroll, jroll) The roll in the i and j dimensions, respectively.

    Returns:
        the rolled array
    """
    (iroll, jroll) = shape
    # The ascontiguousarray bit didn't used to be necessary.  But starting with
    # numpy v1.12, np.roll doesn't seem to always return a C-contiguous array.
    return np.ascontiguousarray(np.roll(np.roll(image, jroll, axis=1), iroll, axis=0))

def kxky(array_shape=(256, 256)):
    """Return the tuple (kx, ky) corresponding to the DFT of a unit integer-sampled array of
    input shape.

    Uses the FFTW conventions for Fourier space, so k varies in approximate range (-pi, pi],
    and the (0, 0) array element corresponds to (kx, ky) = (0, 0).

    See also the docstring for np.fftfreq, which uses the same DFT convention, and is called here,
    but misses a factor of pi.

    Adopts NumPy array index ordering so that the trailing axis corresponds to kx, rather than
    the leading axis as would be expected in IDL/Fortran.  See docstring for ``numpy.meshgrid``
    which also uses this convention.

    Parameters:
        array_shape:    The NumPy array shape desired for kx, ky.

    Returns:
        kx, ky
    """
    # Note: numpy shape is y,x
    k_xaxis = np.fft.fftfreq(array_shape[1]) * 2. * np.pi
    k_yaxis = np.fft.fftfreq(array_shape[0]) * 2. * np.pi
    return np.meshgrid(k_xaxis, k_yaxis)

def g1g2_to_e1e2(g1, g2):
    """Convenience function for going from (g1, g2) -> (e1, e2).

    Here g1 and g2 are reduced shears, and e1 and e2 are distortions - see `Shear`
    for definitions of reduced shear and distortion in terms of axis ratios or other ways of
    specifying ellipses.

    Parameters:
        g1:     First reduced shear component
        g2:     Second reduced shear component

    Returns:
        the corresponding distortions, e1 and e2.
    """
    # Conversion:
    # e = (a^2-b^2) / (a^2+b^2)
    # g = (a-b) / (a+b)
    # b/a = (1-g)/(1+g)
    # e = (1-(b/a)^2) / (1+(b/a)^2)
    gsq = g1*g1 + g2*g2
    if gsq == 0.:
        return 0., 0.
    else:
        g = np.sqrt(gsq)
        boa = (1-g) / (1+g)
        e = (1 - boa*boa) / (1 + boa*boa)
        e1 = g1 * (e/g)
        e2 = g2 * (e/g)
        return e1, e2

def rotate_xy(x, y, theta):
    """Rotates points in the xy-Cartesian plane counter-clockwise through an angle ``theta`` about
    the origin of the Cartesian coordinate system.

    Parameters:
        x:      NumPy array of input x coordinates
        y:      NumPy array of input y coordinates
        theta:  Rotation angle (+ve counter clockwise) as an `Angle` instance

    Returns:
        the rotated coordinates (x_rot,y_rot).
    """
    sint, cost = theta.sincos()
    x_rot = x * cost - y * sint
    y_rot = x * sint + y * cost
    return x_rot, y_rot


class SimpleGenerator:
    """A simple class that is constructed with an arbitrary object.
    Then generator() will return that object.

    This is useful as a way to use an already existing object in a multiprocessing Proxy,
    since that normally needs a factory function.  So this is a factory function that
    just returns an already existing object.
    """
    def __init__(self, obj): self._obj = obj
    def __call__(self):  # pragma: no cover
        return self._obj

def rand_arr(shape, deviate):
    """Function to make a 2d array of random deviates (of any sort).

    Parameters:
        shape:      A list of length 2, indicating the desired 2d array dimensions
        deviate:    Any GalSim deviate (see random.py) such as `UniformDeviate`, `GaussianDeviate`,
                    etc. to be used to generate random numbers

    Returns:
        a NumPy array of the desired dimensions with random numbers generated using the
        supplied deviate.
    """
    tmp = np.empty(tuple(shape), dtype=float)
    deviate.generate(tmp.ravel())
    return tmp

# A helper function for parsing the input position arguments for PowerSpectrum and NFWHalo:
def _convertPositions(pos, units, func):
    """Convert ``pos`` from the valid ways to input positions to two NumPy arrays

    This is used by the functions getShear(), getConvergence(), getMagnification(), and
    getLensing() for both PowerSpectrum and NFWHalo.
    """
    # Check for PositionD or PositionI:
    if isinstance(pos, Position):
        pos = [ pos.x, pos.y ]

    elif len(pos) == 0:
        raise TypeError("Unable to parse the input pos argument for %s."%func)

    # Check for list of Position:
    # The only other options allow pos[0], so if this is invalid, an exception
    # will be raised:
    elif isinstance(pos[0], Position):
        pos = [ np.array([p.x for p in pos], dtype=float),
                np.array([p.y for p in pos], dtype=float) ]

    # Now pos must be a tuple of length 2
    elif len(pos) != 2:
        raise TypeError("Unable to parse the input pos argument for %s."%func)

    else:
        # Check for (x,y):
        try:
            pos = [ float(pos[0]), float(pos[1]) ]
        except TypeError:
            # Only other valid option is ( xlist , ylist )
            pos = [ np.array(pos[0], dtype=float),
                    np.array(pos[1], dtype=float) ]

    # Check validity of units
    if isinstance(units, str):
        # if the string is invalid, this raises a reasonable error message.
        units = AngleUnit.from_name(units)
    if not isinstance(units, AngleUnit):
        raise GalSimValueError("units must be either an AngleUnit or a string", units,
                               ('arcsec', 'arcmin', 'degree', 'hour', 'radian'))

    # Convert pos to arcsec
    if units != arcsec:
        scale = 1. * units / arcsec
        # Note that for the next two lines, pos *must* be a list, not a tuple.  Assignments to
        # elements of tuples is not allowed.
        pos[0] *= scale
        pos[1] *= scale

    return pos

def _spline_approx_err(x, f, left, right, splitpoints, i):
    # For splines, we can't just do the integral over a small range, since the spline slopes
    # are all wrong.  Rather we compute a spline function with the current splitpoints and
    # just the single point in the trial region and recompute a spline function with that.
    # Then we can compute the total error from that approximation.
    # (For the error integral, we still use linear.)

    indices = sorted(splitpoints + [i])
    new_tab = _LookupTable(x[indices], f[indices], 'spline')

    xleft, xright = x[left:i+1], x[i:right+1]
    fleft, fright = f[left:i+1], f[i:right+1]
    f2left = new_tab(xleft)
    f2right = new_tab(xright)
    return trapz(np.abs(fleft-f2left), xleft), trapz(np.abs(fright-f2right), xright)

def _spline_approx_split(x, f, left, right, splitpoints):
    r"""Split a tabulated function into a two-part piecewise spline approximation by exactly
    minimizing \int abs(f(x) - approx(x)) dx.  Operates in O(N^2) time.
    """
    errs = [_spline_approx_err(x, f, left, right, splitpoints, i) for i in range(left+1, right)]
    i = np.argmin(np.sum(errs, axis=1))
    return i+left+1, errs[i]

def _lin_approx_err(x, f, left, right, i):
    r"""Error as \int abs(f(x) - approx(x)) when using ith data point to make piecewise linear
    approximation.
    """
    xleft, xright = x[left:i+1], x[i:right+1]
    fleft, fright = f[left:i+1], f[i:right+1]
    xi, fi = x[i], f[i]
    mleft = (fi-f[left])/(xi-x[left])
    mright = (f[right]-fi)/(x[right]-xi)
    f2left = f[left]+mleft*(xleft-x[left])
    f2right = fi+mright*(xright-xi)
    return trapz(np.abs(fleft-f2left), xleft), trapz(np.abs(fright-f2right), xright)

def _exact_lin_approx_split(x, f, left, right, splitpoints):
    r"""Split a tabulated function into a two-part piecewise linear approximation by exactly
    minimizing \int abs(f(x) - approx(x)) dx.  Operates in O(N^2) time.
    """
    errs = [_lin_approx_err(x, f, left, right, i) for i in range(left+1, right)]
    i = np.argmin(np.sum(errs, axis=1))
    return i+left+1, errs[i]

def _lin_approx_split(x, f, left, right, splitpoints):
    r"""Split a tabulated function into a two-part piecewise linear approximation by approximately
    minimizing \int abs(f(x) - approx(x)) dx.  Chooses the split point by exactly minimizing
    \int (f(x) - approx(x))^2 dx in O(N) time.
    """
    x = x[left:right+1]
    f = f[left:right+1]
    dx = x[2:] - x[:-2]
    # Error contribution on the left.
    ff0 = f[1:-1]-f[0]  # Only need to search between j=1..(N-1)
    xx0 = x[1:-1]-x[0]
    mleft = ff0/xx0  # slope
    errleft = (np.cumsum(dx*ff0**2)
               - 2*mleft*np.cumsum(dx*ff0*xx0)
               + mleft**2*np.cumsum(dx*xx0**2))
    # Error contribution on the right.
    dx = dx[::-1]  # Reversed so that np.cumsum effectively works right-to-left.
    ffN = f[-2:0:-1]-f[-1]
    xxN = x[-2:0:-1]-x[-1]
    mright = ffN/xxN
    errright = (np.cumsum(dx*ffN**2)
                - 2*mright*np.cumsum(dx*ffN*xxN)
                + mright**2*np.cumsum(dx*xxN**2))
    errright = errright[::-1]

    # Get absolute error for the found point.
    i = np.argmin(errleft+errright)
    return i+left+1, _lin_approx_err(x, f, 0, len(x)-1, i+1)

def thin_tabulated_values(x, f, rel_err=1.e-4, trim_zeros=True, preserve_range=True,
                          fast_search=True, interpolant='linear'):
    """
    Remove items from a set of tabulated f(x) values so that the error in the integral is still
    accurate to a given relative accuracy.

    The input ``x`` and ``f`` values can be lists, NumPy arrays, or really anything that can be
    converted to a NumPy array.  The new lists will be output as numpy arrays.

    Parameters:
        x:              The ``x`` values in the f(x) tabulation.
        f:              The ``f`` values in the f(x) tabulation.
        rel_err:        The maximum relative error to allow in the integral from the removal.
                        [default: 1.e-4]
        trim_zeros:     Remove redundant leading and trailing points where f=0?  (The last
                        leading point with f=0 and the first trailing point with f=0 will be
                        retained).  Note that if both trim_leading_zeros and preserve_range are
                        True, then the only the range of ``x`` *after* zero trimming is preserved.
                        [default: True]
        preserve_range: Should the original range of ``x`` be preserved? (True) Or should the ends
                        be trimmed to include only the region where the integral is
                        significant? (False)  [default: True]
        fast_search:    If set to True, then the underlying algorithm will use a relatively fast
                        O(N) algorithm to select points to include in the thinned approximation.
                        If set to False, then a slower O(N^2) algorithm will be used.  We have
                        found that the slower algorithm tends to yield a thinned representation
                        that retains fewer samples while still meeting the relative error
                        requirement.  [default: True]
        interpolant:    The interpolant to assume for the tabulated values. [default: 'linear']

    Returns:
        a tuple of lists (x_new, y_new) with the thinned tabulation.
    """
    if interpolant == 'spline':
        split_fn = _spline_approx_split
    elif fast_search:
        split_fn = _lin_approx_split
    else:
        split_fn = _exact_lin_approx_split

    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)

    # Check for valid inputs
    if len(x) != len(f):
        raise GalSimIncompatibleValuesError("len(x) != len(f)", x=x, f=f)
    if rel_err <= 0 or rel_err >= 1:
        raise GalSimRangeError("rel_err must be between 0 and 1", rel_err, 0., 1.)
    if not (np.diff(x) >= 0).all():
        raise GalSimValueError("input x is not sorted.", x)

    # Check for trivial noop.
    if len(x) <= 2:
        # Nothing to do
        return x,f

    total_integ = trapz(abs(f), x, interpolant)
    if total_integ == 0:
        return np.array([ x[0], x[-1] ]), np.array([ f[0], f[-1] ])
    thresh = total_integ * rel_err

    if trim_zeros:
        first = max(f.nonzero()[0][0]-1, 0)  # -1 to keep one non-redundant zero.
        last = min(f.nonzero()[0][-1]+1, len(x)-1)  # +1 to keep one non-redundant zero.
        x, f = x[first:last+1], f[first:last+1]

    if not preserve_range:
        # Remove values from the front that integrate to less than thresh.
        err_integ1 = 0.5 * (abs(f[0]) + abs(f[1])) * (x[1] - x[0])
        k0 = 0
        x_range = x[-1] - x[0]
        while k0 < len(x)-2 and err_integ1 < thresh * (x[k0+1]-x[0]) / x_range:
            k0 = k0+1
            err_integ1 += 0.5 * (abs(f[k0]) + abs(f[k0+1])) * (x[k0+1] - x[k0])
        # Now the integral from 0 to k0+1 (inclusive) is a bit too large.
        # That means k0 is the largest value we can use that will work as the starting value.

        # Remove values from the back that integrate to less than thresh.
        k1 = len(x)-1
        err_integ2 = 0.5 * (abs(f[k1-1]) + abs(f[k1])) * (x[k1] - x[k1-1])
        while k1 > k0 and err_integ2 < thresh * (x[-1]-x[k1-1]) / x_range:
            k1 = k1-1
            err_integ2 += 0.5 * (abs(f[k1-1]) + abs(f[k1])) * (x[k1] - x[k1-1])
        # Now the integral from k1-1 to len(x)-1 (inclusive) is a bit too large.
        # That means k1 is the smallest value we can use that will work as the ending value.

        # Subtract the error so far from thresh
        if interpolant == 'spline':
            new_integ = trapz(abs(f[k0:k1+1]),x[k0:k1+1], interpolant='spline')
            thresh -= np.abs(new_integ-total_integ)
        else:
            thresh -= trapz(abs(f[:k0]),x[:k0]) + trapz(abs(f[k1:]),x[k1:])

        x = x[k0:k1+1]  # +1 since end of range is given as one-past-the-end.
        f = f[k0:k1+1]

    # Check again for noop after trimming endpoints.
    if len(x) <= 2:
        return x,f

    # Thin interior points.  Start with no interior points and then greedily add them back in one at
    # a time until relative error goal is met.
    # Use a heap to track:
    heap = [(-2*thresh,  # -err; initialize large enough to trigger while loop below.
             0,          # first index of interval
             len(x)-1)]  # last index of interval
    splitpoints = [0,len(x)-1]
    while len(heap) > 0:
        _, left, right = heapq.heappop(heap)
        i, (errleft, errright) = split_fn(x, f, left, right, splitpoints)
        splitpoints.append(i)
        if i > left+1:
            heapq.heappush(heap, (-errleft, left, i))
        if right > i+1:
            heapq.heappush(heap, (-errright, i, right))
        if interpolant != 'spline':
            # This is a sufficient stopping criterion for linear
            if (-sum(h[0] for h in heap) < thresh):
                break
        else:
            # For spline, we also need to recompute the total integral to make sure
            # that the realized total error is less than thresh.
            if (-sum(h[0] for h in heap) < thresh):
                splitpoints = sorted(splitpoints)
                current_integ = trapz(f[splitpoints], x[splitpoints], interpolant)
                if np.abs(current_integ - total_integ) < thresh:
                    break
    splitpoints = sorted(splitpoints)
    return x[splitpoints], f[splitpoints]

def old_thin_tabulated_values(x, f, rel_err=1.e-4, preserve_range=False): # pragma: no cover
    """
    Remove items from a set of tabulated f(x) values so that the error in the integral is still
    accurate to a given relative accuracy.

    The input ``x`` and ``f`` values can be lists, NumPy arrays, or really anything that can be
    converted to a NumPy array.  The new lists will be output as python lists.

    .. note::
        In Issue #739, Josh wrote `thin_tabulated_values`  as a replacement for this function,
        which had been buggy -- not actually hitting its target relative accuracy.  So on the
        same issue, Mike fixed this algorithm to at least work correctly.

        However, we recommend using the above algorithm, since it keeps fewer sample locations
        for a given ``rel_err`` than the old algorithm.

        On the other hand, the old algorithm (this one) may be quite a bit faster, so we retain
        it here in case there is a use case where it is more appropriate.

    Parameters:
        x:              The ``x`` values in the f(x) tabulation.
        f:              The ``f`` values in the f(x) tabulation.
        rel_err:        The maximum relative error to allow in the integral from the removal.
                        [default: 1.e-4]
        preserve_range: Should the original range of ``x`` be preserved? (True) Or should the ends
                        be trimmed to include only the region where the integral is
                        significant? (False)  [default: False]

    Returns:
        a tuple of lists (x_new, y_new) with the thinned tabulation.
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)

    # Check for valid inputs
    if len(x) != len(f):
        raise GalSimIncompatibleValuesError("len(x) != len(f)", x=x, f=f)
    if rel_err <= 0 or rel_err >= 1:
        raise GalSimRangeError("rel_err must be between 0 and 1", rel_err, 0., 1.)
    if not (np.diff(x) >= 0).all():
        raise GalSimValueError("input x is not sorted.", x)

    # Check for trivial noop.
    if len(x) <= 2:
        # Nothing to do
        return x,f

    # Start by calculating the complete integral of |f|
    total_integ = trapz(abs(f),x)
    if total_integ == 0:
        return np.array([ x[0], x[-1] ]), np.array([ f[0], f[-1] ])
    thresh = rel_err * total_integ
    x_range = x[-1] - x[0]

    if not preserve_range:
        # Remove values from the front that integrate to less than thresh.
        err_integ1 = 0.5 * (abs(f[0]) + abs(f[1])) * (x[1] - x[0])
        k0 = 0
        while k0 < len(x)-2 and err_integ1 < thresh * (x[k0+1]-x[0]) / x_range:
            k0 = k0+1
            err_integ1 += 0.5 * (abs(f[k0]) + abs(f[k0+1])) * (x[k0+1] - x[k0])
        # Now the integral from 0 to k0+1 (inclusive) is a bit too large.
        # That means k0 is the largest value we can use that will work as the starting value.

        # Remove values from the back that integrate to less than thresh.
        k1 = len(x)-1
        err_integ2 = 0.5 * (abs(f[k1-1]) + abs(f[k1])) * (x[k1] - x[k1-1])
        while k1 > k0 and err_integ2 < thresh * (x[-1]-x[k1-1]) / x_range:
            k1 = k1-1
            err_integ2 += 0.5 * (abs(f[k1-1]) + abs(f[k1])) * (x[k1] - x[k1-1])
        # Now the integral from k1-1 to len(x)-1 (inclusive) is a bit too large.
        # That means k1 is the smallest value we can use that will work as the ending value.

        # Subtract the error so far from thresh
        thresh -= trapz(abs(f[:k0]),x[:k0]) + trapz(abs(f[k1:]),x[k1:])

        x = x[k0:k1+1]  # +1 since end of range is given as one-past-the-end.
        f = f[k0:k1+1]

        # And update x_range for the new values
        x_range = x[-1] - x[0]

    # Start a new list with just the first item so far
    newx = [ x[0] ]
    newf = [ f[0] ]

    k0 = 0  # The last item currently in the new array
    k1 = 1  # The current item we are considering to skip or include
    while k1 < len(x)-1:
        # We are considering replacing all the true values between k0 and k1+1 (non-inclusive)
        # with a linear approxmation based on the points at k0 and k1+1.
        lin_f = f[k0] + (f[k1+1]-f[k0])/(x[k1+1]-x[k0]) * (x[k0:k1+2] - x[k0])
        # Integrate | f(x) - lin_f(x) | from k0 to k1+1, inclusive.
        err_integ = trapz(np.abs(f[k0:k1+2] - lin_f), x[k0:k1+2])
        # If the integral of the difference is < thresh * (dx/x_range), we can skip this item.
        if abs(err_integ) < thresh * (x[k1+1]-x[k0]) / x_range:
            # OK to skip item k1
            k1 = k1 + 1
        else:
            # Also ok to keep if its own relative error is less than rel_err
            true_integ = trapz(f[k0:k1+2], x[k0:k1+2])
            if abs(err_integ) < rel_err * abs(true_integ):
                # OK to skip item k1
                k1 = k1 + 1
            else:
                # Have to include this one.
                newx.append(x[k1])
                newf.append(f[k1])
                k0 = k1
                k1 = k1 + 1

    # Always include the last item
    newx.append(x[-1])
    newf.append(f[-1])

    return newx, newf


def horner(x, coef, dtype=None):
    """Evaluate univariate polynomial using Horner's method.

    I.e., take A + Bx + Cx^2 + Dx^3 and evaluate it as
    A + x(B + x(C + x(D)))

    Parameters:
        x:      A numpy array of values at which to evaluate the polynomial.
        coef:   Polynomial coefficients of increasing powers of x.
        dtype:  Optionally specify the dtype of the return array. [default: None]

    Returns:
        a numpy array of the evaluated polynomial.  Will be the same shape as x.
    """
    if dtype is None:
        dtype = np.result_type(
            np.min_scalar_type(x),
            np.min_scalar_type(coef)
        )
    result = np.empty_like(x, dtype=dtype)
    # Make sure everything is an array
    if result.dtype == float:
        # And if the result is float, it's worth making sure x, coef are also float and
        # contiguous, so we can use the faster c++ implementation.
        x = np.ascontiguousarray(x, dtype=float)
        coef = np.ascontiguousarray(coef, dtype=float)
    else:
        x = np.asarray(x)
        coef = np.asarray(coef)
    if len(coef.shape) != 1:
        raise GalSimValueError("coef must be 1-dimensional", coef)
    _horner(x, coef, result)
    return result

def _horner(x, coef, result):
    """Equivalent to `horner`, but ``x``, ``coef``, and ``result`` must be contiguous arrays.

    In particular, ``result`` must be already allocated as an array in which to put the answer.
    This is the thing that is returned from the regular `horner`.

    Parameters:
        x:      A numpy array of values at which to evaluate the polynomial.
        coef:   Polynomial coefficients of increasing powers of x.
        result: Numpy array into which to write the result.  Must be same shape as x.
    """
    if result.dtype == float:
        _x = x.__array_interface__['data'][0]
        _coef = coef.__array_interface__['data'][0]
        _result = result.__array_interface__['data'][0]
        _galsim.Horner(_x, x.size, _coef, coef.size, _result)
    else:
        coef = np.trim_zeros(coef, trim='b')  # trim only from the back
        if len(coef) == 0:
            result.fill(0)
            return
        result.fill(coef[-1])
        for c in coef[-2::-1]:
            result *= x
            if c != 0: result += c

def horner2d(x, y, coefs, dtype=None, triangle=False):
    """Evaluate bivariate polynomial using nested Horner's method.

    Parameters:
        x:          A numpy array of the x values at which to evaluate the polynomial.
        y:          A numpy array of the y values at which to evaluate the polynomial.
        coefs:      2D array-like of coefficients in increasing powers of x and y.
                    The first axis corresponds to increasing the power of y, and the second to
                    increasing the power of x.
        dtype:      Optionally specify the dtype of the return array. [default: None]
        triangle:   If True, then the coefs are only non-zero in the upper-left triangle
                    of the array. [default: False]

    Returns:
        a numpy array of the evaluated polynomial.  Will be the same shape as x and y.
    """
    if dtype is None:
        dtype = np.result_type(
            np.min_scalar_type(x),
            np.min_scalar_type(y),
            np.min_scalar_type(coefs)
        )
    result = np.empty_like(x, dtype=dtype)
    temp = np.empty_like(x, dtype=dtype)
    # Make sure everything is an array
    if result.dtype == float:
        # And if the result is float, it's worth making sure x, coef are also float,
        # so we can use the faster c++ implementation.
        x = np.ascontiguousarray(x, dtype=float)
        y = np.ascontiguousarray(y, dtype=float)
        coefs = np.ascontiguousarray(coefs, dtype=float)
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        coefs = np.asarray(coefs)

    if x.shape != y.shape:
        raise GalSimIncompatibleValuesError("x and y are not the same shape", x=x, y=y)
    if len(coefs.shape) != 2:
        raise GalSimValueError("coefs must be 2-dimensional", coefs)
    if triangle and coefs.shape[0] != coefs.shape[1]:
        raise GalSimIncompatibleValuesError("coefs must be square if triangle is True",
                                            coefs=coefs, triangle=triangle)
    _horner2d(x, y, coefs, result, temp, triangle)
    return result

def _horner2d(x, y, coefs, result, temp, triangle=False):
    """Equivalent to `horner2d`, but ``x``, ``y``, ``coefs``, ``result``, and ``temp``
    must be contiguous arrays.

    In particular, ``result`` must be already allocated as an array in which to put the answer.
    This is the thing that is returned from the regular `horner`.  In addition, ``temp`` must
    be allocated for the function to use as temporary work space.

    Parameters:
        x:          A numpy array of the x values at which to evaluate the polynomial.
        y:          A numpy array of the y values at which to evaluate the polynomial.
        coefs:      2D array-like of coefficients in increasing powers of x and y.
                    The first axis corresponds to increasing the power of y, and the second to
                    increasing the power of x.
        result:     Numpy array into which to write the result.  Must be same shape as x.
        temp:       Numpy array to hold temporary results.  Must be the same shape as x.
        triangle:   If True, then the coefs are only non-zero in the upper-left triangle
                    of the array. [default: False]
    """
    if result.dtype == float:
        # Note: the c++ implementation doesn't need to care about triangle.
        # It is able to trivially account for the zeros without special handling.
        _x = x.__array_interface__['data'][0]
        _y = y.__array_interface__['data'][0]
        _coefs = coefs.__array_interface__['data'][0]
        _result = result.__array_interface__['data'][0]
        _temp = temp.__array_interface__['data'][0]
        _galsim.Horner2D(_x, _y, x.size, _coefs, coefs.shape[0], coefs.shape[1], _result, _temp)
    else:
        if triangle:
            result.fill(coefs[-1][0])
            for k, coef in enumerate(coefs[-2::-1]):
                result *= x
                _horner(y, coef[:k+2], temp)
                result += temp
        else:
            _horner(y, coefs[-1], result)
            for coef in coefs[-2::-1]:
                result *= x
                _horner(y, coef, temp)
                result += temp


def horner3d(x, y, u, coefs):
    result = horner2d(y, u, coefs[-1])
    for coef in coefs[-2::-1]:
        result *= x
        result += horner2d(y, u, coef)
    return result


def horner4d(x, y, u, v, coefs):
    result = horner3d(y, u, v, coefs[-1])
    for coef in coefs[-2::-1]:
        result *= x
        result += horner3d(y, u, v, coef)
    return result


def deInterleaveImage(image, N, conserve_flux=False,suppress_warnings=False):
    """
    The routine to do the opposite of what `interleaveImages` routine does. It generates a
    (uniform) dither sequence of low resolution images from a high resolution image.

    Many pixel level detector effects, such as interpixel capacitance, persistence, charge
    diffusion etc. can be included only on images drawn at the native pixel scale, which happen to
    be undersampled in most cases. Nyquist-sampled images that also include the effects of detector
    non-idealities can be obtained by drawing multiple undersampled images (with the detector
    effects included) that are offset from each other by a fraction of a pixel. If the offsets are
    uniformly spaced, then images can be combined using `interleaveImages` into a Nyquist-sampled
    image.

    Drawing multiple low resolution images of a light profile can be a lot slower than drawing a
    high resolution image of the same profile, even if the total number of pixels is the same. A
    uniformly offset dither sequence can be extracted from a well-resolved image that is drawn by
    convolving the surface brightness profile explicitly with the native pixel response and setting
    a lower sampling scale (or higher sampling rate) using the ``pixel_scale`` argument in
    `GSObject.drawImage` routine and setting the ``method`` parameter to 'no_pixel'.

    Here is an example script using this routine:

    Example::

        >>> n = 2
        >>> gal = galsim.Gaussian(sigma=2.8)
        >>> gal_pix = galsim.Convolve([gal,galsim.Pixel(scale=1.0)])
        >>> img = gal_pix.drawImage(gal_pix,scale=1.0/n,method='no_pixel')
        >>> im_list, offsets = galsim.utilities.deInterleaveImage(img,N=n)
        >>> for im in im_list:
        >>>     im.applyNonlinearity(lambda x: x-0.01*x**2) #detector effects
        >>> img_new = galsim.utilities.interleaveImages(im_list,N=n,offsets)

    Parameters:
        image:              Input image from which lower resolution images are extracted.
        N:                  Number of images extracted in either directions. It can be of type
                            'int' if equal number of images are extracted in both directions or a
                            list or tuple of two integers, containing the number of images in x
                            and y directions respectively.
        conserve_flux:      Should the routine output images that have, on average, same total
                            pixel values as the input image (True) or should the pixel values
                            summed over all the images equal the sum of pixel values of the input
                            image (False)? [default: False]
        suppress_warnings:  Suppresses the warnings about the pixel scale of the output, if True.
                            [default: False]

    Returns:
        a list of images (`Image`) and offsets (`PositionD`) to reconstruct the input image using
        `interleaveImages`.
    """
    if isinstance(N,int):
        n1,n2 = N,N
    else:
        try:
            n1,n2 = N
        except (TypeError, ValueError):
            raise TypeError("N must be an integer or a tuple of two integers") from None

    if not isinstance(image, Image):
        raise TypeError("image must be an instance of galsim.Image")

    y_size,x_size = image.array.shape
    if x_size%n1 or y_size%n2:
        raise GalSimIncompatibleValuesError(
            "The value of N is incompatible with the dimensions of the image to be deinterleaved",
            N=N, image=image)

    im_list, offsets = [], []
    for i in range(n1):
        for j in range(n2):
            # The tricky part - going from array indices to Image coordinates (x,y)
            # DX[i'] = -(i+0.5)/n+0.5 = -i/n + 0.5*(n-1)/n
            #    i  = -n DX[i'] + 0.5*(n-1)
            dx,dy = -(i+0.5)/n1+0.5,-(j+0.5)/n2+0.5
            offset = _PositionD(dx,dy)
            img_arr = image.array[j::n2,i::n1].copy()
            img = Image(img_arr)
            if conserve_flux is True:
                img *= n1*n2
            im_list.append(img)
            offsets.append(offset)

    wcs = image.wcs
    if wcs is not None and wcs._isUniform:
        jac = wcs.jacobian()
        for img in im_list:
            img_wcs = JacobianWCS(jac.dudx*n1,jac.dudy*n2,jac.dvdx*n1,jac.dvdy*n2)
            ## Since pixel scale WCS is not equal to its jacobian, checking if img_wcs is a pixel
            ## scale
            img_wcs_decomp = img_wcs.getDecomposition()
            if img_wcs_decomp[1].g==0:
                img.wcs = PixelScale(img_wcs_decomp[0])
            else:
                img.wcs = img_wcs
            ## Preserve the origin so that the interleaved image has the same bounds as the image
            ## that is being deinterleaved.
            img.setOrigin(image.origin)

    elif suppress_warnings is False:
        galsim_warn("Individual images could not be assigned a WCS automatically.")

    return im_list, offsets


def interleaveImages(im_list, N, offsets, add_flux=True, suppress_warnings=False,
                     catch_offset_errors=True):
    """
    Interleaves the pixel values from two or more images and into a single larger image.

    This routine converts a list of images taken at a series of (uniform) dither offsets into a
    single higher resolution image, where the value in each final pixel is the observed pixel
    value from exactly one of the original images.  It can be used to build a Nyquist-sampled image
    from a set of images that were observed with pixels larger than the Nyquist scale.

    In the original observed images, the integration of the surface brightness over the pixels is
    equivalent to convolution by the pixel profile and then sampling at the centers of the pixels.
    This procedure simulates an observation sampled at a higher resolution than the original images,
    while retaining the original pixel convolution.

    Such an image can be obtained in a fairly simple manner in simulations of surface brightness
    profiles by convolving them explicitly with the native pixel response and setting a lower
    sampling scale (or higher sampling rate) using the ``pixel_scale`` argument in
    `GSObject.drawImage` routine and setting the ``method`` parameter to 'no_pixel'.

    However, pixel level detector effects can be included only on images drawn at the native pixel
    scale, which happen to be undersampled in most cases. Nyquist-sampled images that also include
    the effects of detector non-idealities can be obtained by drawing multiple undersampled images
    (with the detector effects included) that are offset from each other by a fraction of a pixel.

    This is similar to other procedures that build a higher resolution image from a set of low
    resolution images, such as MultiDrizzle and IMCOM. A disadvantage of this routine compared to
    ther others is that the images must be offset in equal steps in each direction. This is
    difficult to acheive with real observations but can be precisely acheived in a series of
    simulated images.

    An advantage of this procedure is that the noise in the final image is not correlated as the
    pixel values are each taken from just a single input image. Thus, this routine preserves the
    noise properties of the pixels.

    Here's an example script using this routine:

    Example::

        >>> n = 2
        >>> gal = galsim.Gaussian(sigma=2.8)
        >>> DX = numpy.arange(0.0,1.0,1./n)
        >>> DX -= DX.mean()
        >>> im_list, offsets = [], []
        >>> for dx in DX:
            ... offset = galsim.PositionD(dx,0.0)
            ... offsets.append(offset)
            ... im = galsim.Image(16,16)
            ... gal.drawImage(image=im,offset=offset,scale=1.0)
            ... im.applyNonlinearity(lambda x: x - 0.01*x**2) # detector effects
            ... im_list.append(im)
        >>> img = galsim.utilities.interleaveImages(im_list=im_list,N=(n,1),offsets=offsets)

    Parameters:
        im_list:                A list containing the `galsim.Image` instances to be interleaved.
        N:                      Number of images to interleave in either directions. It can be of
                                type ``int`` if equal number of images are interleaved in both
                                directions or a list or tuple of two integers, containing the number
                                of images in x and y directions respectively.
        offsets:                A list containing the offsets as galsim.PositionD instances
                                corresponding to every image in ``im_list``. The offsets must be
                                spaced equally and must span an entire pixel area. The offset
                                values must be symmetric around zero, hence taking positive and
                                negative values, with upper and lower limits of +0.5 and -0.5
                                (limit values excluded).
        add_flux:               Should the routine add the fluxes of all the images (True) or
                                average them (False)? [default: True]
        suppress_warnings:      Suppresses the warnings about the pixel scale of the output, if
                                True.  [default: False]
        catch_offset_errors:    Checks for the consistency of ``offsets`` with ``N`` and raises
                                errors if inconsistencies found (True). Recommended, but could slow
                                down the routine a little. [default: True]

    Returns:
        the interleaved `Image`
    """
    if isinstance(N,int):
        n1,n2 = N,N
    else:
        try:
            n1,n2 = N
        except (TypeError, ValueError):
            raise TypeError("N must be an integer or a tuple of two integers") from None

    if len(im_list)<2:
        raise GalSimValueError("im_list must have at least two instances of galsim.Image", im_list)

    if (n1*n2 != len(im_list)):
        raise GalSimIncompatibleValuesError(
            "N is incompatible with the number of images in im_list", N=N, im_list=im_list)

    if len(im_list)!=len(offsets):
        raise GalSimIncompatibleValuesError(
            "im_list and offsets must be lists of same length", im_list=im_list, offsets=offsets)

    for offset in offsets:
        if not isinstance(offset, PositionD):
            raise TypeError("offsets must be a list of galsim.PositionD instances")

    if not isinstance(im_list[0], Image):
        raise TypeError("im_list must be a list of galsim.Image instances")

    # These should be the same for all images in im_list.
    y_size, x_size = im_list[0].array.shape
    wcs = im_list[0].wcs

    for im in im_list[1:]:
        if not isinstance(im, Image):
            raise TypeError("im_list must be a list of galsim.Image instances")

        if im.array.shape != (y_size,x_size):
            raise GalSimIncompatibleValuesError(
                "All galsim.Image instances in im_list must be of the same size", im_list=im_list)

        if im.wcs != wcs:
            raise GalSimIncompatibleValuesError(
                "All galsim.Image instances in im_list must have the same WCS", im_list=im_list)

    img_array = np.zeros((n2*y_size,n1*x_size))
    # The tricky part - going from (x,y) Image coordinates to array indices
    # DX[i'] = -(i+0.5)/n+0.5 = -i/n + 0.5*(n-1)/n
    #    i  = -n DX[i'] + 0.5*(n-1)
    for k in range(len(offsets)):
        dx, dy = offsets[k].x, offsets[k].y

        i = int(round((n1-1)*0.5-n1*dx))
        j = int(round((n2-1)*0.5-n2*dy))

        if catch_offset_errors is True:
            err_i = (n1-1)*0.5-n1*dx - round((n1-1)*0.5-n1*dx)
            err_j = (n2-1)*0.5-n2*dy - round((n2-1)*0.5-n2*dy)
            tol = 1.e-6
            if abs(err_i)>tol or abs(err_j)>tol:
                raise GalSimIncompatibleValuesError(
                    "offsets must be a list of galsim.PositionD instances with x values "
                    "spaced by 1/{0} and y values by 1/{1} around 0.".format(n1,n2),
                    N=N, offsets=offsets)

            if i<0 or j<0 or i>=n1 or j>=n2:
                raise GalSimIncompatibleValuesError(
                    "offsets must be a list of galsim.PositionD instances with x values "
                    "spaced by 1/{0} and y values by 1/{1} around 0.".format(n1,n2),
                    N=N, offsets=offsets)
        else:
            # If we're told to just trust the offsets, at least make sure the slice will be
            # the right shape.
            i = i%n1
            j = j%n2

        img_array[j::n2,i::n1] = im_list[k].array

    img = Image(img_array)
    if not add_flux:
        # Fix the flux normalization
        img /= 1.0*len(im_list)

    # Assign an appropriate WCS for the output
    if wcs is not None and wcs._isUniform:
        jac = wcs.jacobian()
        dudx, dudy, dvdx, dvdy = jac.dudx, jac.dudy, jac.dvdx, jac.dvdy
        img_wcs = JacobianWCS(1.*dudx/n1,1.*dudy/n2,1.*dvdx/n1,1.*dvdy/n2)
        ## Since pixel scale WCS is not equal to its jacobian, checking if img_wcs is a pixel scale
        img_wcs_decomp = img_wcs.getDecomposition()
        if img_wcs_decomp[1].g==0: ## getDecomposition returns scale,shear,angle,flip
            img.wcs = PixelScale(img_wcs_decomp[0])
        else:
            img.wcs = img_wcs

    elif not suppress_warnings:
        galsim_warn("Interleaved image could not be assigned a WCS automatically.")

    # Assign a possibly non-trivial origin and warn if individual image have different origins.
    orig = im_list[0].origin
    img.setOrigin(orig)
    if any(im.origin != orig for im in im_list[1:]):
        if not suppress_warnings:
            galsim_warn("Images in im_list have multiple values for origin. Assigning the "
                        "origin of the first Image instance in im_list to the interleaved image.")

    return img

@contextmanager
def printoptions(*args, **kwargs):
    """A context manager for using different numpy printoptions temporarily

    From http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array

    Usage::

        with printoptions(threshold=len(long_arr)):
            print(long_arr)  # Doesn't omit values in the middle of the array
        print(long_arr)  # If the array is long enough, will use ... in the middle.

    .. note::
        It seems Numpy finally added this feature in version 1.15.  So this is probably
        equivalent to using ``numpy.printoptions`` instead of ``galsim.utilities.printoptions``.
    """
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    # contextmanager exception handling is tricky.  Don't forget to wrap the yield:
    # http://preshing.com/20110920/the-python-with-statement-by-example/
    try:
        yield
    finally:
        np.set_printoptions(**original)


_pickle_shared = False

@contextmanager
def pickle_shared():
    """A context manager to flag that one wishes to include object state from shared memory in
    pickle objects.

    Example::

        obj = galsim_obj_with_shared_state()  # e.g., galsim.phase_screens.AtmosphericScreen
        pickle.dump(obj, file)

        # restart python, unloading shared state
        obj = pickle.load(file)  # fails due to missing shared state.

        obj = galsim_obj_with_shared_state()
        with pickle_shared():
            pickle.dump(obj, filename)

        # restart python, again unloading shared state
        obj = pickle.load(file)  # loads both obj and required shared state.
    """
    global _pickle_shared
    original = _pickle_shared
    _pickle_shared = True
    try:
        yield
    finally:
        _pickle_shared = original


def listify(arg):
    """Turn argument into a list if not already iterable.

    Parameters:
        arg:        An argument that may be a lit or a single item

    Returns:
        [arg] if arg was not already a list, otherwise arg.
    """
    return [arg] if not hasattr(arg, '__iter__') else arg


def dol_to_lod(dol, N=None, scalar_string=True):
    """Generate list of dicts from dict of lists (with broadcasting).
    Specifically, generate "scalar-valued" kwargs dictionaries from a kwarg dictionary with values
    that are length-N lists, or possibly length-1 lists or scalars that should be broadcasted up to
    length-N lists.

    Parameters:
        dol:            A dict of lists
        N:              The length of the lists if known in advance. [default: None, which will
                        determine the maximum length of the lists for you]
        scalar_string:  Whether strings in the input list should be treated as scalars (and thus
                        broadcast to each item in the output) or not (in which case, they will
                        be treated as lists of characters) [default: True]

    Returns:
        A list of dicts
    """
    if N is None:
        if scalar_string:
            lens = [len(v) for v in dol.values()
                           if hasattr(v, '__len__')
                           and not isinstance(v, str)]
        else:
            lens = [len(v) for v in dol.values()
                           if hasattr(v, '__len__')]
        N = max(lens) if lens != [] else 1
    # Loop through broadcast range
    for i in range(N):
        out = {}
        for k, v in dol.items():
            if scalar_string and isinstance(v, str):
                out[k] = v
                continue
            try:
                out[k] = v[i]
            except IndexError:  # It's list-like, but too short.
                if len(v) != 1:
                    raise GalSimIncompatibleValuesError(
                        "Cannot broadcast kwargs of different non-length-1 lengths.", dol=dol) from None
                out[k] = v[0]
            except TypeError:  # Value is not list-like, so broadcast it in its entirety.
                out[k] = v
            except Exception:
                raise GalSimIncompatibleValuesError(
                    "Cannot broadcast kwarg {0}={1}".format(k, v), dol=dol)
        yield out

def structure_function(image):
    r"""Estimate the angularly-averaged structure function of a 2D random field.

    The angularly-averaged structure function D(r) of the 2D field phi is defined as:

    .. math::
        D(|r|) = \langle |phi(x) - phi(x+r)|^2 \rangle

    where the x and r on the RHS are 2D vectors, but the :math:`|r|` on the LHS is just a scalar
    length.

    The image must have its ``scale`` attribute defined.  It will be used in the calculations to
    set the scale of the radial distances.

    Parameters:
        image:  `Image` containing random field realization.

    Returns:
        A python callable mapping a separation length r to the estimate of the structure
        function D(r).
    """
    array = image.array
    ny, nx = array.shape
    scale = image.scale

    # The structure function can be derived from the correlation function B(r) as:
    # D(r) = 2 * [B(0) - B(r)]

    corr = np.fft.ifft2(np.abs(np.fft.fft2(np.fft.fftshift(array)))**2).real / (nx * ny)
    # Check that the zero-lag correlation function is equal to the variance before doing the
    # ifftshift.
    #assert (corr[0, 0] / np.var(array) - 1.0) < 1e-6
    corr = np.fft.ifftshift(corr)

    x = scale * (np.arange(nx) - nx//2)
    y = scale * (np.arange(ny) - ny//2)
    tab = LookupTable2D(x, y, corr)
    thetas = np.arange(0., 2*np.pi, 100)  # Average over these angles.

    return lambda r: 2*(tab(0.0, 0.0) - np.mean(tab(r*np.cos(thetas), r*np.sin(thetas))))

def merge_sorted(arrays):
    r"""Merge 2 or more numpy arrays into a single merged array.

    Each of the input arrays must be already sorted.

    This is equivalent to np.unique(np.concatenate(arrays)), but much faster.

    Parameters:
        arrays:     A list of arrays to merge.

    Returns:
        A single numpy.array with the merged values.
    """
    try:
        return _galsim.MergeSorted(list(arrays))
    except Exception as e:
        # Probably the inputs are not sorted.  Try to give the user more information.
        for i,a in enumerate(arrays):
            if not np.all(np.diff(a)>=0):
                raise GalSimValueError("Not all arrays input to merge_sorted are sorted." +
                                       "The first such case is at index %s"%i,
                                       value=a) from None
        else:
            # That wasn't it.  Just reraise the exception, converted to a GalSimError.
            raise GalSimError(str(e)) from None

def combine_wave_list(*args):
    """Combine wave_list attributes of all objects in obj_list while respecting blue_limit and
    red_limit attributes.  Should work with any combination of `SED`, `Bandpass`, and
    `ChromaticObject` instances.

    Parameters:
        obj_list:   List of `SED`, `Bandpass`, or `ChromaticObject` instances.

    Returns:
        wave_list, blue_limit, red_limit
    """
    if len(args) == 1:
        if isinstance(args[0], (list, tuple)):
            args = args[0]
        else:
            raise TypeError("Single input argument must be a list or tuple")

    if len(args) == 0:
        return np.array([], dtype=float), 0.0, np.inf

    if len(args) == 1:
        obj = args[0]
        return obj.wave_list, getattr(obj, 'blue_limit', 0.0), getattr(obj, 'red_limit', np.inf)

    blue_limit = np.max([getattr(obj, 'blue_limit', 0.0) for obj in args])
    red_limit = np.min([getattr(obj, 'red_limit', np.inf) for obj in args])
    if blue_limit > red_limit:
        raise GalSimError("Empty wave_list intersection.")

    waves = [np.asarray(obj.wave_list) for obj in args]
    waves = [w[(blue_limit <= w) & (w <= red_limit)] for w in waves]
    # Make sure both limits are included in final list
    if any(len(w) > 0 for w in waves):
        waves.append([blue_limit, red_limit])
    wave_list = merge_sorted(waves)

    return wave_list, blue_limit, red_limit

def functionize(f):
    """Decorate a function ``f`` which accepts scalar positional or keyword arguments, to accept
    arguments that can be either scalars or _functions_.  If the arguments include univariate
    (N-variate) functions, then the output will be a univariate (N-variate) function.  While it's
    okay to mix scalar and N-variate function arguments, it is an error to mix N-variate and
    M-variate function arguments.

    Example::

        >>> def f(x, y):      # Function of two scalars.
        ...     return x + y
        >>> decorated = functionize(f)   # Function of two scalars, functions, or a mix.
        >>> result = f(2, 3)  # 5
        >>> result = f(2, lambda u: u)  # Generates a TypeError
        >>> result = decorated(2, 3)  # Scalar args returns a scalar
        >>> result = decorated(2, lambda u: u)  # Univariate argument leads to a univariate output.
        >>> print(result(5))  # 7
        >>> result = decorated(2, lambda u,v: u*v)  # Bivariate argument leads to a bivariate output.
        >>> print(result(5, 7))  # 2 + (5*7) = 37

    We can use arguments that accept keyword arguments too::

        >>> def f2(u, v=None):
        ...    if v is None:
        ...        v = 6.0
        ...    return u / v
        >>> result = decorated(2, f2)
        >>> print(result(12))  # 2 + (12./6) = 4.0
        >>> print(result(12, v=4))  # 2 + (12/4) = 5

    Note that you can also use python's decorator syntax::

        >>> @functionize
        >>> def f(x, y):
        ...     return x + y

    Parameters:
        f:      The function to be decorated.

    Returns:
        The decorated function.
    """
    @functools.wraps(f)
    def ff(*args, **kwargs):
        # First check if any of the arguments are callable...
        if not any(hasattr(arg, '__call__') for arg in args+tuple(kwargs.values())):
            return f(*args, **kwargs)  # ... if not, then keep output type a scalar ...
        else:
            def fff(*inner_args, **inner_kwargs): # ...else the output type is a function: fff.
                new_args = [arg
                            if not hasattr(arg, '__call__')
                            else arg(*inner_args, **inner_kwargs)
                            for arg in args]
                new_kwargs = dict([(k, v)
                                   if not hasattr(v, '__call__')
                                   else (k, v(*inner_args, **inner_kwargs))
                                   for k, v in kwargs.items()])
                return f(*new_args, **new_kwargs)
            return fff
    return ff

def binomial(a, b, n):
    """Return xy coefficients of (ax + by)^n ordered by descending powers of a.

    Example::

        # (x + y)^3 = 1 x^3 + 3 x^2 y + 3 x y^2 + 1 y^3
        >>>  print(binomial(1, 1, 3))
        array([ 1.,  3.,  3.,  1.])


        # (2 x + y)^3 = 8 x^3 + 12 x^2 y + 6 x y^2 + 1 y^3
        >>>  print(binomial(2, 1, 3))
        array([ 8.,  12.,  6.,  1.])

    Parameters:
        a:      First scalar in binomial to be expanded.
        b:      Second scalar in binomial to be expanded.
        n:      Exponent of expansion.

    Returns:
        Array of coefficients in expansion.
    """
    b_over_a = float(b)/float(a)
    def generate():
        c = a**n
        yield c
        for i in range(n):  # pragma: no branch  (It never actually gets past the last yield.)
            c *= b_over_a * (n-i)/(i+1)
            yield c
    return np.fromiter(generate(), float, n+1)

def unweighted_moments(image, origin=None):
    """Computes unweighted 0th, 1st, and 2nd moments in image coordinates.  Respects image bounds,
    but ignores any scale or wcs.

    Parameters:
        image:      `Image` from which to compute moments
        origin:     Optional origin in image coordinates used to compute Mx and My
                    [default: galsim.PositionD(0, 0)].

    Returns:
        Dict with entries for [M0, Mx, My, Mxx, Myy, Mxy]
    """
    if origin is None:
        origin = _PositionD(0,0)
    a = image.array.astype(float)
    offset = image.origin - origin
    xgrid, ygrid = np.meshgrid(np.arange(image.array.shape[1]) + offset.x,
                               np.arange(image.array.shape[0]) + offset.y)
    M0 = np.sum(a)
    Mx = np.sum(xgrid * a) / M0
    My = np.sum(ygrid * a) / M0
    Mxx = np.sum(((xgrid-Mx)**2) * a) / M0
    Myy = np.sum(((ygrid-My)**2) * a) / M0
    Mxy = np.sum((xgrid-Mx) * (ygrid-My) * a) / M0
    return dict(M0=M0, Mx=Mx, My=My, Mxx=Mxx, Myy=Myy, Mxy=Mxy)

def unweighted_shape(arg):
    """Computes unweighted second moment size and ellipticity given either an image or a dict of
    unweighted moments.

    The size is:
        rsqr = Mxx+Myy
    The ellipticities are:
        e1 = (Mxx-Myy) / rsqr
        e2 = 2*Mxy / rsqr

    Parameters:
        arg:    Either a `galsim.Image` or the output of unweighted_moments(image).

    Returns:
        Dict with entries for [rsqr, e1, e2]
    """
    if isinstance(arg, Image):
        arg = unweighted_moments(arg)
    rsqr = arg['Mxx'] + arg['Myy']
    return dict(rsqr=rsqr, e1=(arg['Mxx']-arg['Myy'])/rsqr, e2=2*arg['Mxy']/rsqr)

def rand_with_replacement(n, n_choices, rng, weight=None, _n_rng_calls=False):
    """Select some number of random choices from a list, with replacement, using a supplied RNG.

    ``n`` random choices with replacement are made assuming that those choices should range from 0
    to ``n_choices-1``, so they can be used as indices in a list/array.  If ``weight`` is supplied,
    then it should be an array of length ``n_choices`` that ranges from 0-1, and can be used to
    make weighted choices from the list.

    Parameters:
        n:          Number of random selections to make.
        n_choices:  Number of entries from which to choose.
        rng:        RNG to use.  Must a `galsim.BaseDeviate` instance.
        weight:     Optional list of weight factors to use for weighting the selection of
                    random indices.

    Returns:
        a NumPy array of length n containing the integer-valued indices that were selected.
    """
    # Make sure we got a proper RNG.
    if not isinstance(rng, BaseDeviate):
        raise TypeError("The rng provided to rand_with_replacement() must be a BaseDeviate")
    ud = UniformDeviate(rng)

    # Sanity check the requested number of random indices.
    # Note: we do not require that the type be an int, as long as the value is consistent with
    # an integer value (i.e., it could be a float 1.0 or 1).
    if n != int(n) or n < 1:
        raise GalSimValueError("n must be an integer >= 1.", n)
    if n_choices != int(n_choices) or n_choices < 1:
        raise GalSimValueError("n_choices must be an integer >= 1.", n_choices)

    # Sanity check the input weight.
    if weight is not None:
        # We need some sanity checks here in case people passed in weird values.
        if len(weight) != n_choices:
            raise GalSimIncompatibleValuesError(
                "Array of weights has wrong length", weight=weight, n_choices=n_choices)
        if (np.any(np.isnan(weight)) or np.any(np.isinf(weight)) or
            np.min(weight)<0 or np.max(weight)>1):
            raise GalSimRangeError("Supplied weights include values outside [0,1] or inf/NaN.",
                                   weight, 0., 1.)

    # We first make a random list of integer indices.
    index = np.zeros(n)
    ud.generate(index)
    if _n_rng_calls:
        # Here we use the undocumented kwarg (for internal use by config) to track the number of
        # RNG calls.
        n_rng_calls = n
    index = (n_choices*index).astype(int)

    # Then we account for the weights, if possible.
    if weight is not None:
        # If weight factors are available, make sure the random selection uses the weights.
        test_vals = np.zeros(n)
        # Note that the weight values by definition have a maximum of 1, as enforced above.
        ud.generate(test_vals)
        if _n_rng_calls:
            n_rng_calls += n
        # The ones with mask==True are the ones we should replace.
        mask = test_vals > weight[index]
        while np.any(mask):
            # Update the index and test values for those that failed. We have to do this by
            # generating random numbers into new arrays, because ud.generate() does not enable
            # us to directly populate a sub-array like index[mask] or test_vals[mask].
            n_fail = mask.astype(int).sum()
            # First update the indices that failed.
            new_arr = np.zeros(n_fail)
            ud.generate(new_arr)
            index[mask] = (n_choices*new_arr).astype(int)
            # Then update the test values that failed.
            new_test_vals = np.zeros(n_fail)
            ud.generate(new_test_vals)
            test_vals[mask] = new_test_vals
            if _n_rng_calls:
                n_rng_calls += 2*n_fail
            # Finally, update the test array used to determine whether any galaxies failed.
            mask = test_vals > weight[index]

    if _n_rng_calls:
        return index, n_rng_calls
    else:
        return index


def check_share_file(filename, subdir):
    """Find `SED` or `Bandpass` file, possibly adding share_dir/subdir.

    Parameters:
        filename:       The file name to look for
        subdir:         The subdirectory of `galsim.meta_data.share_dir` where this file might be.

    Returns:
        True, correct_filename      if the file was found
        False, ''                   if not
    """
    if os.path.isfile(filename):
        return True, filename

    new_filename = os.path.join(meta_data.share_dir, subdir, filename)
    if os.path.isfile(new_filename):
        return True, new_filename
    else:
        return False, ''


class OrderedWeakRef(weakref.ref):
    """Assign an arbitrary ordering to weakref.ref so that it can be part of a heap.
    """
    def __lt__(self, other):
        return id(self) < id(other)


def nCr(n, r):
    """Combinations.  I.e., the number of ways to choose ``r`` distiguishable things from ``n``
    distinguishable things.

    Parameters:
        n       The number of options to choose from.
        r       The number of items to choose

    Returns:
        nCr, the (n,r) binomial coefficient.

    .. note::
        In Python 3, the factorial function was improved such that doing this the direct way
        of calculating n! / (r! (n-r)!) seems to be the fastest algorith.  In Python 2, for
        largish values of n, a more complicated algorithm that avoided large integers was
        faster.  This function uses the direct method for both -- we don't bother to check the
        version of Python to potentially select a different algorithm in the two cases.
    """
    if 0 <= r <= n:
        return math.factorial(n) // (math.factorial(r) * math.factorial(n-r))
    else:
        return 0

def set_omp_threads(num_threads, logger=None):
    """Set the number of OpenMP threads to use in the C++ layer.

    :param num_threads: The target number of threads to use (If None or <=0, then try to use the
                        numer of cpus.)
    :param logger:      If desired, a logger object for logging any warnings here. (default: None)

    :returns:           The  number of threads OpenMP reports that it will use.  Typically this
                        matches the input, but OpenMP reserves the right not to comply with
                        the requested number of threads.
    """
    # This function was copied shamelessly from TreeCorr's function of the same name.

    input_num_threads = num_threads  # Save the input value.

    # If num_threads is auto, get it from cpu_count
    if num_threads is None or num_threads <= 0:
        num_threads = multiprocessing.cpu_count()
        if logger:
            logger.debug('multiprocessing.cpu_count() = %d',num_threads)

    # Tell OpenMP to use this many threads
    if logger:
        logger.debug('Telling OpenMP to use %d threads',num_threads)

    # Cf. comment in get_omp_threads.  Do it here too.
    var = "OMP_PROC_BIND"
    if var not in os.environ:
        os.environ[var] = "false"
    num_threads = _galsim.SetOMPThreads(num_threads)

    # Report back appropriately.
    if logger:
        logger.debug('OpenMP reports that it will use %d threads',num_threads)
        if num_threads > 1:
            logger.info('Using %d threads.',num_threads)
        elif input_num_threads is not None and input_num_threads != 1:
            # Only warn if the user specifically asked for num_threads != 1.
            logger.warning("Unable to use multiple threads, since OpenMP is not enabled.")

    return num_threads

def get_omp_threads():
    """Get the current number of OpenMP threads to be used in the C++ layer.

    :returns: num_threads
    """
    # Some OMP implemenations have a bug where if omp_get_max_threads() is called
    # (which is what this function does), it sets something called thread affinity.
    # The upshot of that is that multiprocessing (i.e. not even just omp threading) is confined
    # to a single hardware thread.  Yeah, it's idiotic, but that seems to be the case.
    # The only solution found by Eli, who looked into it pretty hard, is to set the env
    # variable OMP_PROC_BIND to "false".  This seems to stop the bad behavior.
    # So we do it here always before calling GetOMPThreads.
    # If this breaks someone valid use of this variable, let us know and we can try to
    # come up with another solution, but without this lots of multiprocessing breaks.
    var = "OMP_PROC_BIND"
    if var not in os.environ:
        os.environ[var] = "false"
    return _galsim.GetOMPThreads()

@contextmanager
def single_threaded(*, num_threads=1):
    """A context manager that turns off (or down) OpenMP threading e.g. during multiprocessing.

    Usage:

    .. code::

        with single_threaded():
            # Code where you don't want to use any OpenMP threads in the C++ layer
            # E.g. starting a multiprocessing pool, where you don't want each process
            # to use multiple threads, potentially ending up with n_cpu^2 threads
            # running at once, which would generally be bad for performance.

    .. note::

        This is especaily important when your compiler is gcc and you are using the
        "fork" context in multiprocessing.  There is a bug in gcc that can cause an
        OpenMP parallel block to hang after forking.
        cf. `make it possible to use OMP on both sides of a fork <https://patchwork.ozlabs.org/project/gcc/patch/CAPJVwBkOF5GnrMr=4d1ehEKRGkY0tHzJzfXAYBguawu9y5wxXw@mail.gmail.com/#712883>`_
        for more discussion about this issue.

    It can also be used to set a particular number of threads other than 1, using the
    optional parameter ``num_threads``, although the original intent of this class is
    to leave that as 1 (the default).

    Parameters:
        num_threads:    The number of threads you want during the context [default: 1]
    """
    # Get the current number of threads here, so we can set it back when we're done.
    orig_num_threads = get_omp_threads()
    temp_num_threads = num_threads

    # If threadpoolctl is installed, use that too, since it will set blas libraries to
    # be single threaded too. This makes it so you don't need to set the environment
    # variables OPENBLAS_NUM_THREAD=1 or MKL_NUM_THREADS=1, etc.
    try:
        import threadpoolctl
    except ImportError:
        tpl = None
    else:  # pragma: no cover  (Not installed on GHA currently.)
        tpl = threadpoolctl.threadpool_limits(num_threads)

    set_omp_threads(temp_num_threads)
    with warnings.catch_warnings():
        # Starting in python 3.12, there is a deprecation warning about using fork when
        # a process is multithreaded.  Unfortunately, this applies even to processes that
        # are currently single threaded, but used multi-threading previously.
        # So if a user is doing something in an explicitly single-threaded context,
        # suppress this DeprecationWarning.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield

    set_omp_threads(orig_num_threads)
    if tpl is not None:  # pragma: no cover
        tpl.unregister()



# The rest of these are only used by the tests in GalSim.  But we make them available
# for other code bases who may want to use them as well.

def check_pickle(obj, func = lambda x : x, irreprable=False, random=None):
    """Check that the object is picklable.

    Also check some related desirable properties that we want for (almost) all galsim objects:

    1. pickle.loads(pickle.dumps(obj)) recovers something equivalent to the original.
    2. obj != object() and object() != obj.  (I.e. it doesn't == things it shouldn't.)
    3. hash(obj) is the same for two equivalent objects, if it is hashable.
    4. copy.copy(obj) returns an equivalent copy unless random=True.
    5. copy.deepcopy(obj) returns an equivalent copy.
    6. eval(repr(obj)) returns an equivalent copy unless random or irreprable=True.
    7. repr(obj) makes something if irreprable=False.

    Parameters:
        obj:            The object to test
        func:           A function of obj to use to test for equivalence.  [default: lambda x: x]
        irreprable:     Whether to skip the eval(repr(obj)) test.  [default: False]
        random:         Whether the obj has some intrinsic randomness. [default: False, unless
                        it has an rng attribute or it is a galsim.BaseDeviate]
    """
    # In case the repr uses these:
    import galsim
    import coord
    import astropy
    from numpy import array, uint16, uint32, int16, int32, float32, float64, complex64, complex128, ndarray
    from astropy.units import Unit
    import astropy.io.fits

    print('Try pickling ',str(obj))

    obj2 = pickle.loads(pickle.dumps(obj))
    assert obj2 is not obj
    f1 = func(obj)
    f2 = func(obj2)
    assert f2 == f1, f"func(obj) = {f1}\nfunc(obj2) = {f2}\nrepr(obj) = {repr(obj)}\nrepr(obj2)={repr(obj2)}"

    # Check that == works properly if the other thing isn't the same type.
    assert f1 != object()
    assert object() != f1

    # Test the hash values are equal for two equivalent objects.
    if isinstance(obj, Hashable):
        assert hash(obj) == hash(obj2), f"hash(obj) = {hash(obj)}, hash(obj2) = {hash(obj2)}"

    obj3 = copy.copy(obj)
    assert obj3 is not obj
    if random is None:
        random = hasattr(obj, 'rng') or isinstance(obj, BaseDeviate) or 'rng' in repr(obj)
    if not random:  # Things with an rng attribute won't be identical on copy.
        f3 = func(obj3)
        assert f3 == f1
    elif isinstance(obj, BaseDeviate):
        f1 = func(obj)  # But BaseDeviates will be ok.  Just need to remake f1.
        f3 = func(obj3)
        assert f3 == f1, f"func(obj) = {f1}\nfunc(obj3) = {f3}"

    obj4 = copy.deepcopy(obj)
    assert obj4 is not obj
    f4 = func(obj4)
    if random: f1 = func(obj)
    # But everything should be identical with deepcopy.
    assert f4 == f1, f"func(obj) = {f1}\nfunc(obj4) = {f4}"

    # Also test that the repr is an accurate representation of the object.
    # The gold standard is that eval(repr(obj)) == obj.  So check that here as well.
    # A few objects we don't expect to work this way in GalSim; when testing these, we set the
    # `irreprable` kwarg to true.  Also, we skip anything with random deviates since these don't
    # respect the eval/repr roundtrip.

    if not random and not irreprable:
        # A further complication is that the default numpy print options do not lead to sufficient
        # precision for the eval string to exactly reproduce the original object, and start
        # truncating the output for relatively small size arrays.  So we temporarily bump up the
        # precision and truncation threshold for testing.
        with printoptions(precision=20, threshold=np.inf):
            obj5 = eval(repr(obj))
        f5 = func(obj5)
        assert f5 == f1, f"func(obj) = {f1}\nfunc(obj5) = {f5}\nrepr(obj) = {repr(obj)}\nrepr(obj5)={repr(obj5)}"
    else:
        # Even if we're not actually doing the test, still make the repr to check for syntax errors.
        repr(obj)

    # Historical note:
    # We used to have a test here where we perturbed the construction arguments to make sure
    # that objects that should be different really are different.  However, that used
    # `__getinitargs__`, which we don't use anymore, so we removed this section.
    # This means that this inequality test has to be done manually via check_all_diff.
    # See releases v2.3 or earlier for the old way we did this.


def check_all_diff(objs, check_hash=True):
    """Test that all objects test as being unequal.

    It checks all pairs of objects in the list and asserts that obj1 != obj2.

    If check_hash=True, then it also checks that their hashes are different.

    Parameters:
        objs:           A list of objects to test.
        check_hash:     Whether to also check the hash values.
    """
    # Check that all objects are unique.
    # Would like to use `assert len(objs) == len(set(objs))` here, but this requires that the
    # elements of objs are hashable (and that they have unique hashes!, which is what we're trying
    # to test!.  So instead, we just loop over all combinations.
    for i, obji in enumerate(objs):
        assert obji == obji
        assert not (obji != obji)
        # Could probably start the next loop at `i+1`, but we start at 0 for completeness
        # (and to verify a != b implies b != a)
        for j, objj in enumerate(objs):
            if i == j:
                continue
            assert obji != objj, ("Found equivalent objects {0} == {1} at indices {2} and {3}"
                                  .format(obji, objj, i, j))

    if not check_hash:
        return
    # Now check that all hashes are unique (if the items are hashable).
    if not isinstance(objs[0], Hashable):
        return
    hashes = [hash(obj) for obj in objs]
    if not (len(hashes) == len(set(hashes))):  # pragma: no cover
        for k, v in Counter(hashes).items():
            if v <= 1:
                continue
            print("Found multiple equivalent object hashes:")
            for i, obj in enumerate(objs):
                if hash(obj) == k:
                    print(i, repr(obj))
    assert len(hashes) == len(set(hashes))


def timer(f):
    """A decorator that reports how long a function took to run.

    In GalSim we decorate all of our tests with this to try to watch for long-running tests.
    """
    @functools.wraps(f)
    def f2(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2


class CaptureLog:
    """A context manager that saves logging output into a string that is accessible for
    checking in unit tests.

    After exiting the context, the attribute ``output`` will have the logging output.

    Sample usage:

            >>> with CaptureLog() as cl:
            ...     cl.logger.info('Do some stuff')
            >>> assert cl.output == 'Do some stuff'

    """
    def __init__(self, level=3):
        logging_levels = { 0: logging.CRITICAL,
                           1: logging.WARNING,
                           2: logging.INFO,
                           3: logging.DEBUG }
        self.logger = logging.getLogger('CaptureLog')
        self.logger.setLevel(logging_levels[level])
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.addHandler(self.handler)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.handler.flush()
        self.output = self.stream.getvalue().strip()
        self.handler.close()


# Context to make it easier to profile bits of the code
class Profile:
    """A context manager that profiles a snippet of code.

    Sample usage:

            >>> with Profile():
            ...     slow_function()

    Parameters:
        sortby:     What parameter to sort the results by.  [default: tottime]
        nlines:     How many lines of output to report.  [default: 30]
        reverse:    Whether to reverse the order of the output lines to put the most important
                    items at the bottom rather than the top. [default: False]
        filename:   If desired, a file to output the full profile information in pstats format.
                    [default: None]
    """
    def __init__(self, sortby='tottime', nlines=30, reverse=False, filename=None):
        self.sortby = sortby
        self.nlines = nlines
        self.reverse = reverse
        self.filename = filename

    def __enter__(self):
        self.pr = cProfile.Profile()
        self.pr.enable()
        return self

    def __exit__(self, type, value, traceback):
        self.pr.disable()
        if self.filename:  # pragma: no cover
            self.pr.dump_stats(self.filename)
        ps = pstats.Stats(self.pr).sort_stats(self.sortby)
        if self.reverse:  # pragma: no cover
            ps = ps.reverse_order()
        ps.print_stats(self.nlines)


def least_squares(fun, x0, args=(), kwargs={}, max_iter=1000, tol=1e-9, lambda_init=1.0):
    """Perform a non-linear least squares fit using the Levenberg-Marquardt algorithm.

    Drop in replacement for scipy.optimize.least_squares when using default options,
    though many fewer options available in general.

    Parameters:
        fun: Function which computes vector of residuals, with the signature
             fun(params, *args, **kwargs) -> np.ndarray.
        x0: Initial guess for the parameters.
        args: Additional arguments to pass to the function.
        kwargs: Additional keyword arguments to pass to the function.
        max_iter: Maximum number of iterations.  [default: 1000]
        tol: Tolerance for convergence.  [default: 1e-9]
        lambda_init: Initial damping factor for Levenberg-Marquardt.  [default: 1.0]

    Returns:
        A named tuple with fields:
            x: The final parameter values.
            cost: The final cost (sum of squared residuals).
    """
    # JM: This is a tweaked version of a ChatGPT-generated implementation of
    # Levenberg-Marquardt (cross-checked against the wikipedia page
    # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm).

    from collections import namedtuple
    params = np.array(x0)
    lambda_ = lambda_init

    for _ in range(max_iter):  # pragma: no branch
        residuals = fun(params, *args, **kwargs)

        # Jacobian matrix
        J = np.zeros((len(residuals), len(params)))
        for j in range(len(params)):
            perturbation = np.zeros(len(params))
            perturbation[j] = tol
            J[:, j] = (fun(params + perturbation, *args, **kwargs) - residuals) / tol

        # Regular least squares param update
        JTJ = np.dot(J.T, J)
        JTr = np.dot(J.T, residuals)

        # Levenberg-Marquardt adjustment
        A = JTJ + lambda_ * np.eye(len(JTJ))
        try:
            delta_params = np.linalg.solve(A, JTr)
        except np.linalg.LinAlgError:
            lambda_ *= 2
            continue

        new_params = params - delta_params
        new_residuals = fun(new_params, *args, **kwargs)

        if np.linalg.norm(new_residuals) < np.linalg.norm(residuals):
            params = new_params
            residuals = new_residuals
            lambda_ /= 3  # reduce damping
        else:
            lambda_ *= 3  # increase damping

        if np.linalg.norm(delta_params) < tol:
            break

    cost = 0.5 * np.sum(residuals**2)

    # Create a result object similar to scipy.optimize.OptimizeResult
    Result = namedtuple('Result', ['x', 'cost'])
    result = Result(x=params, cost=cost)

    return result
