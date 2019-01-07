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
"""@file utilities.py
Module containing general utilities for the GalSim software.
"""
import functools
from contextlib import contextmanager
from future.utils import iteritems
from builtins import range, object
import weakref
import os
import numpy as np

from . import _galsim
from .errors import GalSimError, GalSimValueError, GalSimIncompatibleValuesError, GalSimRangeError
from .errors import galsim_warn


def roll2d(image, shape):
    """Perform a 2D roll (circular shift) on a supplied 2D NumPy array, conveniently.

    @param image        The NumPy array to be circular shifted.
    @param shape        (iroll, jroll) The roll in the i and j dimensions, respectively.

    @returns the rolled image.
    """
    (iroll, jroll) = shape
    # The ascontiguousarray bit didn't used to be necessary.  But starting with
    # numpy v1.12, np.roll doesn't seem to always return a C-contiguous array.
    return np.ascontiguousarray(np.roll(np.roll(image, jroll, axis=1), iroll, axis=0))

def kxky(array_shape=(256, 256)):
    """Return the tuple `(kx, ky)` corresponding to the DFT of a unit integer-sampled array of input
    shape.

    Uses the SBProfile conventions for Fourier space, so `k` varies in approximate range (-pi, pi].
    Uses the most common DFT element ordering conventions (and those of FFTW), so that `(0, 0)`
    array element corresponds to `(kx, ky) = (0, 0)`.

    See also the docstring for np.fftfreq, which uses the same DFT convention, and is called here,
    but misses a factor of pi.

    Adopts NumPy array index ordering so that the trailing axis corresponds to `kx`, rather than the
    leading axis as would be expected in IDL/Fortran.  See docstring for numpy.meshgrid which also
    uses this convention.

    @param array_shape   The NumPy array shape desired for `kx, ky`.
    """
    # Note: numpy shape is y,x
    k_xaxis = np.fft.fftfreq(array_shape[1]) * 2. * np.pi
    k_yaxis = np.fft.fftfreq(array_shape[0]) * 2. * np.pi
    return np.meshgrid(k_xaxis, k_yaxis)

def g1g2_to_e1e2(g1, g2):
    """Convenience function for going from `(g1, g2)` -> `(e1, e2)`.

    Here `g1` and `g2` are reduced shears, and `e1` and `e2` are distortions - see shear.py for
    definitions of reduced shear and distortion in terms of axis ratios or other ways of specifying
    ellipses.

    @param g1   First reduced shear component
    @param g2   Second reduced shear component

    @returns the corresponding distortions, e1 and e2.
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
    """Rotates points in the xy-Cartesian plane counter-clockwise through an angle `theta` about the
    origin of the Cartesian coordinate system.

    @param x        NumPy array of input `x` coordinates
    @param y        NumPy array of input `y` coordinates
    @param theta    Rotation angle (+ve counter clockwise) as an Angle instance

    @return the rotated coordinates `(x_rot,y_rot)`.
    """
    sint, cost = theta.sincos()
    x_rot = x * cost - y * sint
    y_rot = x * sint + y * cost
    return x_rot, y_rot

def parse_pos_args(args, kwargs, name1, name2, integer=False, others=[]):
    """Parse the args and kwargs of a function call to be some kind of position.

    We allow four options:

        f(x,y)
        f(galsim.PositionD(x,y)) or f(galsim.PositionI(x,y))
        f( (x,y) )  (or any indexable thing)
        f(name1=x, name2=y)

    If the inputs must be integers, set `integer=True`.
    If there are other args/kwargs to parse after these, then their names should be
    be given as the parameter `others`, which are passed back in a tuple after the position.
    """
    from .position import PositionD, PositionI
    def canindex(arg):
        try: arg[0], arg[1]
        except (TypeError, IndexError): return False
        else: return True

    other_vals = []
    if len(args) == 0:
        # Then name1,name2 need to be kwargs
        try:
            x = kwargs.pop(name1)
            y = kwargs.pop(name2)
        except KeyError:
            raise TypeError('Expecting kwargs %s, %s.  Got %s'%(name1, name2, kwargs.keys()))
    elif ( ( isinstance(args[0], PositionI) or
             (not integer and isinstance(args[0], PositionD)) ) and
           len(args) <= 1+len(others) ):
        x = args[0].x
        y = args[0].y
        for arg in args[1:]:
            other_vals.append(arg)
            others.pop(0)
    elif canindex(args[0]) and len(args) <= 1+len(others):
        x = args[0][0]
        y = args[0][1]
        for arg in args[1:]:
            other_vals.append(arg)
            others.pop(0)
    elif len(args) == 1:
        if integer:
            raise TypeError("Cannot parse argument %s as a PositionI"%(args[0]))
        else:
            raise TypeError("Cannot parse argument %s as a PositionD"%(args[0]))
    elif len(args) <= 2 + len(others):
        x = args[0]
        y = args[1]
        for arg in args[2:]:
            other_vals.append(arg)
            others.pop(0)
    else:
        raise TypeError("Too many arguments supplied")
    # Read any remaining other kwargs
    if others:
        for name in others:
            val = kwargs.pop(name)
            other_vals.append(val)
    if kwargs:
        raise TypeError("Received unexpected keyword arguments: %s",kwargs)

    if integer:
        pos = PositionI(int(x),int(y))
    else:
        pos = PositionD(float(x),float(y))
    if other_vals:
        return (pos,) + tuple(other_vals)
    else:
        return pos


class SimpleGenerator:
    """A simple class that is constructed with an arbitrary object.
    Then generator() will return that object.
    """
    def __init__(self, obj): self._obj = obj
    def __call__(self):  # pragma: no cover  (It is covered, but coveralls doesn't get it right.)
        return self._obj

class AttributeDict(object): # pragma: no cover
    """Dictionary class that allows for easy initialization and refs to key values via attributes.

    NOTE: Modified a little from Jim's bot.git AttributeDict class so that tab completion now works
    in ipython since attributes are actually added to __dict__.

    HOWEVER this means the __dict__ attribute has been redefined to be a collections.defaultdict()
    so that Jim's previous default attribute behaviour is also replicated.
    """
    def __init__(self):
        import collections
        object.__setattr__(self, "__dict__", collections.defaultdict(AttributeDict))

    def __getattr__(self, name):
        return self.__dict__[name]

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def merge(self, other):
        self.__dict__.update(other.__dict__)

    def _write(self, output, prefix=""):
        for k, v in iteritems(self.__dict__):
            if isinstance(v, AttributeDict):
                v._write(output, prefix="{0}{1}.".format(prefix, k))
            else:
                output.append("{0}{1} = {2}".format(prefix, k, repr(v)))

    def __bool__(self):
        return not not self.__dict__

    def __repr__(self):
        output = []
        self._write(output, "")
        return "\n".join(output)

    __str__ = __repr__

    def __len__(self):
        return len(self.__dict__)

def rand_arr(shape, deviate):
    """Function to make a 2d array of random deviates (of any sort).

    @param shape        A list of length 2, indicating the desired 2d array dimensions
    @param deviate      Any GalSim deviate (see random.py) such as UniformDeviate, GaussianDeviate,
                        etc. to be used to generate random numbers

    @returns a NumPy array of the desired dimensions with random numbers generated using the
    supplied deviate.
    """
    from .image import ImageD
    from .noise import DeviateNoise
    tmp = np.empty(tuple(shape), dtype=float)
    deviate.generate(tmp.ravel())
    return tmp

def convert_interpolant(interpolant):
    """Convert a given interpolant to an Interpolant if it is given as a string.
    """
    from .interpolant import Interpolant
    if isinstance(interpolant, Interpolant):
        return interpolant
    else:
        # Will raise an appropriate exception if this is invalid.
        return Interpolant.from_name(interpolant)

# A helper function for parsing the input position arguments for PowerSpectrum and NFWHalo:
def _convertPositions(pos, units, func):
    """Convert `pos` from the valid ways to input positions to two NumPy arrays

       This is used by the functions getShear(), getConvergence(), getMagnification(), and
       getLensing() for both PowerSpectrum and NFWHalo.
    """
    from .position import Position
    from .angle import AngleUnit, arcsec
    # Check for PositionD or PositionI:
    if isinstance(pos, Position):
        pos = [ pos.x, pos.y ]

    elif len(pos) == 0:
        raise TypeError("Unable to parse the input pos argument for %s."%func)

    # Check for list of Position:
    # The only other options allow pos[0], so if this is invalid, an exception
    # will be raised:
    elif isinstance(pos[0], Position):
        pos = [ np.array([p.x for p in pos], dtype='float'),
                np.array([p.y for p in pos], dtype='float') ]

    # Now pos must be a tuple of length 2
    elif len(pos) != 2:
        raise TypeError("Unable to parse the input pos argument for %s."%func)

    else:
        # Check for (x,y):
        try:
            pos = [ float(pos[0]), float(pos[1]) ]
        except TypeError:
            # Only other valid option is ( xlist , ylist )
            pos = [ np.array(pos[0], dtype='float'),
                    np.array(pos[1], dtype='float') ]

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

def _lin_approx_err(x, f, i):
    r"""Error as \int abs(f(x) - approx(x)) when using ith data point to make piecewise linear
    approximation."""
    xleft, xright = x[:i+1], x[i:]
    fleft, fright = f[:i+1], f[i:]
    xi, fi = x[i], f[i]
    mleft = (fi-f[0])/(xi-x[0])
    mright = (f[-1]-fi)/(x[-1]-xi)
    f2left = f[0]+mleft*(xleft-x[0])
    f2right = fi+mright*(xright-xi)
    return np.trapz(np.abs(fleft-f2left), xleft), np.trapz(np.abs(fright-f2right), xright)

def _exact_lin_approx_split(x, f):
    r"""Split a tabulated function into a two-part piecewise linear approximation by exactly
    minimizing \int abs(f(x) - approx(x)) dx.  Operates in O(N^2) time.
    """
    errs = [_lin_approx_err(x, f, i) for i in range(1, len(x)-1)]
    i = np.argmin(np.sum(errs, axis=1))
    return i+1, errs[i]

def _lin_approx_split(x, f):
    r"""Split a tabulated function into a two-part piecewise linear approximation by approximately
    minimizing \int abs(f(x) - approx(x)) dx.  Chooses the split point by exactly minimizing
    \int (f(x) - approx(x))^2 dx in O(N) time.
    """
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
    return i+1, _lin_approx_err(x, f, i+1)

def thin_tabulated_values(x, f, rel_err=1.e-4, trim_zeros=True, preserve_range=True,
                          fast_search=True):
    """
    Remove items from a set of tabulated f(x) values so that the error in the integral is still
    accurate to a given relative accuracy.

    The input `x,f` values can be lists, NumPy arrays, or really anything that can be converted
    to a NumPy array.  The new lists will be output as numpy arrays.

    @param x                The `x` values in the f(x) tabulation.
    @param f                The `f` values in the f(x) tabulation.
    @param rel_err          The maximum relative error to allow in the integral from the removal.
                            [default: 1.e-4]
    @param trim_zeros       Remove redundant leading and trailing points where f=0?  (The last
                            leading point with f=0 and the first trailing point with f=0 will be
                            retained).  Note that if both trim_leading_zeros and preserve_range are
                            True, then the only the range of `x` *after* zero trimming is preserved.
                            [default: True]
    @param preserve_range   Should the original range of `x` be preserved? (True) Or should the ends
                            be trimmed to include only the region where the integral is
                            significant? (False)  [default: True]
    @param fast_search      If set to True, then the underlying algorithm will use a relatively fast
                            O(N) algorithm to select points to include in the thinned approximation.
                            If set to False, then a slower O(N^2) algorithm will be used.  We have
                            found that the slower algorithm tends to yield a thinned representation
                            that retains fewer samples while still meeting the relative error
                            requirement.  [default: True]

    @returns a tuple of lists `(x_new, y_new)` with the thinned tabulation.
    """
    from heapq import heappush, heappop

    split_fn = _lin_approx_split if fast_search else _exact_lin_approx_split

    x = np.array(x)
    f = np.array(f)

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

    total_integ = np.trapz(abs(f), x)
    if total_integ == 0:
        return np.array([ x[0], x[-1] ]), np.array([ f[0], f[-1] ])
    thresh = total_integ * rel_err

    if trim_zeros:
        first = max(f.nonzero()[0][0]-1, 0)  # -1 to keep one non-redundant zero.
        last = min(f.nonzero()[0][-1]+1, len(x)-1)  # +1 to keep one non-redundant zero.
        x, f = x[first:last+1], f[first:last+1]

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
        thresh -= np.trapz(abs(f[:k0]),x[:k0]) + np.trapz(abs(f[k1:]),x[k1:])

        x = x[k0:k1+1]  # +1 since end of range is given as one-past-the-end.
        f = f[k0:k1+1]

        # And update x_range for the new values
        x_range = x[-1] - x[0]

    # Check again for noop after trimming endpoints.
    if len(x) <= 2:
        return x,f

    # Thin interior points.  Start with no interior points and then greedily add them back in one at
    # a time until relative error goal is met.
    # Use a heap to track:
    heap = [(-2*thresh,  # -err; initialize large enough to trigger while loop below.
             0,          # first index of interval
             len(x)-1)]  # last index of interval
    while (-sum(h[0] for h in heap) > thresh):
        _, left, right = heappop(heap)
        i, (errleft, errright) = split_fn(x[left:right+1], f[left:right+1])
        heappush(heap, (-errleft, left, i+left))
        heappush(heap, (-errright, i+left, right))
    splitpoints = sorted([0]+[h[2] for h in heap])
    return x[splitpoints], f[splitpoints]


# In Issue #739, Josh wrote the above algorithm as a replacement for the one here.
# It had been buggy, not actually hitting its target relative accuracy, so on the same issue,
# Mike fixed this algorithm to at least work correctly.  However, we recommend using the above
# algorithm, since it keeps fewer sample locations for a given rel_err than the old algorithm.
# On the other hand, the old algorithm can be quite a bit faster, being O(N), not O(N^2), so
# we retain the old algorithm here in case we want to re-enable it for certain applications.
def old_thin_tabulated_values(x, f, rel_err=1.e-4, preserve_range=False): # pragma: no cover
    """
    Remove items from a set of tabulated f(x) values so that the error in the integral is still
    accurate to a given relative accuracy.

    The input `x,f` values can be lists, NumPy arrays, or really anything that can be converted
    to a NumPy array.  The new lists will be output as python lists.

    @param x                The `x` values in the f(x) tabulation.
    @param f                The `f` values in the f(x) tabulation.
    @param rel_err          The maximum relative error to allow in the integral from the removal.
                            [default: 1.e-4]
    @param preserve_range   Should the original range of `x` be preserved? (True) Or should the ends
                            be trimmed to include only the region where the integral is
                            significant? (False)  [default: False]

    @returns a tuple of lists `(x_new, y_new)` with the thinned tabulation.
    """
    x = np.array(x)
    f = np.array(f)

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
    total_integ = np.trapz(abs(f),x)
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
        thresh -= np.trapz(abs(f[:k0]),x[:k0]) + np.trapz(abs(f[k1:]),x[k1:])

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
        err_integ = np.trapz(np.abs(f[k0:k1+2] - lin_f), x[k0:k1+2])
        # If the integral of the difference is < thresh * (dx/x_range), we can skip this item.
        if abs(err_integ) < thresh * (x[k1+1]-x[k0]) / x_range:
            # OK to skip item k1
            k1 = k1 + 1
        else:
            # Also ok to keep if its own relative error is less than rel_err
            true_integ = np.trapz(f[k0:k1+2], x[k0:k1+2])
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

    @param x        A numpy array of values at which to evaluate the polynomial.
    @param coef     Polynomial coefficients of increasing powers of x.
    @param dtype    Optionally specify the dtype of the return array. [default: None]

    @returns a numpy array of the evaluated polynomial.  Will be the same shape as x.
    """
    result = np.empty_like(x, dtype=dtype)
    # Make sure everything is an array
    if result.dtype == float:
        # And if the result is float, it's worth making sure x, coef are also float and
        # contiguous, so we can use the faster c++ implementation.
        x = np.ascontiguousarray(x, dtype=float)
        coef = np.ascontiguousarray(coef, dtype=float)
    else:
        x = np.array(x, copy=False)
        coef = np.array(coef, copy=False)
    if len(coef.shape) != 1:
        raise GalSimValueError("coef must be 1-dimensional", coef)
    _horner(x, coef, result)
    return result

def _horner(x, coef, result):
    if result.dtype == float:
        _galsim.Horner(x.ctypes.data, x.size, coef.ctypes.data, coef.size, result.ctypes.data)
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

    @param x        A numpy array of the x values at which to evaluate the polynomial.
    @param y        A numpy array of the y values at which to evaluate the polynomial.
    @param coefs    2D array-like of coefficients in increasing powers of x and y.
                    The first axis corresponds to increasing the power of y, and the second to
                    increasing the power of x.
    @param dtype    Optionally specify the dtype of the return array. [default: None]
    @param triangle If True, then the coefs are only non-zero in the upper-left triangle
                    of the array. [default: False]

    @returns a numpy array of the evaluated polynomial.  Will be the same shape as x and y.
    """
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
        x = np.array(x, copy=False)
        y = np.array(y, copy=False)
        coefs = np.array(coefs, copy=False)

    if len(x) != len(y):
        raise GalSimIncompatibleValuesError("x and y are not the same size",
                                            x=x, y=y)
    if len(coefs.shape) != 2:
        raise GalSimValueError("coefs must be 2-dimensional", coefs)
    if triangle and coefs.shape[0] != coefs.shape[1]:
        raise GalSimIncompatibleValuesError("coefs must be square if triangle is True",
                                            coefs=coefs, triangle=triangle)
    _horner2d(x, y, coefs, result, temp, triangle)
    return result

def _horner2d(x, y, coefs, result, temp, triangle=False):
    if result.dtype == float:
        # Note: the c++ implementation doesn't need to care about triangle.
        # It is able to trivially account for the zeros without special handling.
        _galsim.Horner2D(x.ctypes.data, y.ctypes.data, x.size,
                         coefs.ctypes.data, coefs.shape[0], coefs.shape[1],
                         result.ctypes.data, temp.ctypes.data)
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


def deInterleaveImage(image, N, conserve_flux=False,suppress_warnings=False):
    """
    The routine to do the opposite of what 'interleaveImages' routine does. It generates a
    (uniform) dither sequence of low resolution images from a high resolution image.

    Many pixel level detector effects, such as interpixel capacitance, persistence, charge
    diffusion etc. can be included only on images drawn at the native pixel scale, which happen to
    be undersampled in most cases. Nyquist-sampled images that also include the effects of detector
    non-idealities can be obtained by drawing multiple undersampled images (with the detector
    effects included) that are offset from each other by a fraction of a pixel. If the offsets are
    uniformly spaced, then images can be combined using 'interleaveImages' into a Nyquist-sampled
    image.

    Drawing multiple low resolution images of a light profile can be a lot slower than drawing a
    high resolution image of the same profile, even if the total number of pixels is the same. A
    uniformly offset dither sequence can be extracted from a well-resolved image that is drawn by
    convolving the surface brightness profile explicitly with the native pixel response and setting
    a lower sampling scale (or higher sampling rate) using the `pixel_scale' argument in drawImage()
    routine and setting the `method' parameter to `no_pixel'.

    Here is an example script using this routine:

    Interleaving four Gaussian images
    ---------------------------------

        >>> n = 2
        >>> gal = galsim.Gaussian(sigma=2.8)
        >>> gal_pix = galsim.Convolve([gal,galsim.Pixel(scale=1.0)])
        >>> img = gal_pix.drawImage(gal_pix,scale=1.0/n,method='no_pixel')
        >>> im_list, offsets = galsim.utilities.deInterleaveImage(img,N=n)
        >>> for im in im_list:
        >>>     im.applyNonlinearity(lambda x: x-0.01*x**2) #detector effects
        >>> img_new = galsim.utilities.interleaveImages(im_list,N=n,offsets)

    @param image             Input image from which lower resolution images are extracted.
    @param N                 Number of images extracted in either directions. It can be of type
                             'int' if equal number of images are extracted in both directions or a
                             list or tuple of two integers, containing the number of images in x
                             and y directions respectively.
    @param conserve_flux     Should the routine output images that have, on average, same total
                             pixel values as the input image (True) or should the pixel values
                             summed over all the images equal the sum of pixel values of the input
                             image (False)? [default: False]
    @param suppress_warnings Suppresses the warnings about the pixel scale of the output, if True.
                             [default: False]

    @returns a list of images and offsets to reconstruct the input image using 'interleaveImages'.
    """
    from .image import Image
    from .position import PositionD
    from .wcs import JacobianWCS, PixelScale
    if isinstance(N,int):
        n1,n2 = N,N
    else:
        try:
            n1,n2 = N
        except (TypeError, ValueError):
            raise TypeError("N must be an integer or a tuple of two integers")

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
            offset = PositionD(dx,dy)
            img_arr = image.array[j::n2,i::n1].copy()
            img = Image(img_arr)
            if conserve_flux is True:
                img *= n1*n2
            im_list.append(img)
            offsets.append(offset)

    wcs = image.wcs
    if wcs is not None and wcs.isUniform():
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
    sampling scale (or higher sampling rate) using the `pixel_scale' argument in drawImage()
    routine and setting the `method' parameter to `no_pixel'.

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

    Interleaving two Gaussian images along the x-axis
    -------------------------------------------------

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

    @param im_list             A list containing the galsim.Image instances to be interleaved.
    @param N                   Number of images to interleave in either directions. It can be of
                               type `int' if equal number of images are interleaved in both
                               directions or a list or tuple of two integers, containing the number
                               of images in x and y directions respectively.
    @param offsets             A list containing the offsets as galsim.PositionD instances
                               corresponding to every image in `im_list'. The offsets must be spaced
                               equally and must span an entire pixel area. The offset values must
                               be symmetric around zero, hence taking positive and negative values,
                               with upper and lower limits of +0.5 and -0.5 (limit values excluded).
    @param add_flux            Should the routine add the fluxes of all the images (True) or average
                               them (False)? [default: True]
    @param suppress_warnings   Suppresses the warnings about the pixel scale of the output, if True.
                               [default: False]
    @param catch_offset_errors Checks for the consistency of `offsets` with `N` and raises Errors
                               if inconsistencies found (True). Recommended, but could slow down
                               the routine a little. [default: True]

    @returns the interleaved image
    """
    from .position import PositionD
    from .image import Image
    from .wcs import PixelScale, JacobianWCS
    if isinstance(N,int):
        n1,n2 = N,N
    else:
        try:
            n1,n2 = N
        except (TypeError, ValueError):
            raise TypeError("N must be an integer or a tuple of two integers")

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

    # These should be the same for all images in `im_list'.
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
    if wcs is not None and wcs.isUniform():
        jac = wcs.jacobian()
        dudx, dudy, dvdx, dvdy = jac.dudx, jac.dudy, jac.dvdx, jac.dvdy
        img_wcs = JacobianWCS(1.*dudx/n1,1.*dudy/n2,1.*dvdx/n1,1.*dvdy/n2)
        ## Since pixel scale WCS is not equal to its jacobian, checking if img_wcs is a pixel scale
        img_wcs_decomp = img_wcs.getDecomposition()
        if img_wcs_decomp[1].g==0: ## getDecomposition returns scale,shear,angle,flip
            img.wcs = PixelScale(img_wcs_decomp[0])
        else:
            img.wcs = img_wcs

    elif suppress_warnings is False:
        galsim_warn("Interleaved image could not be assigned a WCS automatically.")

    # Assign a possibly non-trivial origin and warn if individual image have different origins.
    orig = im_list[0].origin
    img.setOrigin(orig)
    for im in im_list[1:]:
        if not im.origin==orig:  # pragma: no cover
            galsim_warn("Images in `im_list' have multiple values for origin. Assigning the "
                        "origin of the first Image instance in 'im_list' to the interleaved image.")
            break

    return img

class LRU_Cache:
    """ Simplified Least Recently Used Cache.
    Mostly stolen from http://code.activestate.com/recipes/577970-simplified-lru-cache/,
    but added a method for dynamic resizing.  The least recently used cached item is
    overwritten on a cache miss.

    @param user_function   A python function to cache.
    @param maxsize         Maximum number of inputs to cache.  [Default: 1024]

    Usage
    -----
    >>> def slow_function(*args) # A slow-to-evaluate python function
    >>>    ...
    >>>
    >>> v1 = slow_function(*k1)  # Calling function is slow
    >>> v1 = slow_function(*k1)  # Calling again with same args is still slow
    >>> cache = galsim.utilities.LRU_Cache(slow_function)
    >>> v1 = cache(*k1)  # Returns slow_function(*k1), slowly the first time
    >>> v1 = cache(*k1)  # Returns slow_function(*k1) again, but fast this time.

    Methods
    -------
    >>> cache.resize(maxsize) # Resize the cache, either upwards or downwards.  Upwards resizing
                              # is non-destructive.  Downwards resizing will remove the least
                              # recently used items first.
    """
    def __init__(self, user_function, maxsize=1024):
        # Link layout:     [PREV, NEXT, KEY, RESULT]
        self.root = root = [None, None, None, None]
        self.user_function = user_function
        self.cache = cache = {}

        last = root
        for i in range(maxsize):
            key = object()
            cache[key] = last[1] = last = [last, root, key, None]
        root[0] = last

    def __call__(self, *key):
        cache = self.cache
        root = self.root
        link = cache.get(key)
        if link is not None:
            # Cache hit: move link to last position
            link_prev, link_next, _, result = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            last = root[0]
            last[1] = root[0] = link
            link[0] = last
            link[1] = root
            return result
        # Cache miss: evaluate and insert new key/value at root, then increment root
        #             so that just-evaluated value is in last position.
        result = self.user_function(*key)
        root = self.root  # re-establish root in case user_function modified it due to recursion
        root[2] = key
        root[3] = result
        oldroot = root
        root = self.root = root[1]
        root[2], oldkey = None, root[2]
        root[3], oldvalue = None, root[3]
        del cache[oldkey]
        cache[key] = oldroot
        return result

    def resize(self, maxsize):
        """ Resize the cache.  Increasing the size of the cache is non-destructive, i.e.,
        previously cached inputs remain in the cache.  Decreasing the size of the cache will
        necessarily remove items from the cache if the cache is already filled.  Items are removed
        in least recently used order.

        @param maxsize  The new maximum number of inputs to cache.
        """
        oldsize = len(self.cache)
        if maxsize == oldsize:
            return
        else:
            root = self.root
            cache = self.cache
            if maxsize <= 0:
                raise GalSimValueError("Invalid maxsize", maxsize)
            if maxsize < oldsize:
                for i in range(oldsize - maxsize):
                    # Delete root.next
                    current_next_link = root[1]
                    new_next_link = root[1] = root[1][1]
                    new_next_link[0] = root
                    del cache[current_next_link[2]]
            else: #  maxsize > oldsize:
                for i in range(maxsize - oldsize):
                    # Insert between root and root.next
                    key = object()
                    cache[key] = link = [root, root[1], key, None]
                    root[1][0] = link
                    root[1] = link


# http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
@contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    # contextmanager exception handling is tricky.  Don't forget to wrap the yield:
    # http://preshing.com/20110920/the-python-with-statement-by-example/
    try:
        yield
    finally:
        np.set_printoptions(**original)


def listify(arg):
    """Turn argument into a list if not already iterable."""
    return [arg] if not hasattr(arg, '__iter__') else arg


def dol_to_lod(dol, N=None):
    """Generate list of dicts from dict of lists (with broadcasting).
    Specifically, generate "scalar-valued" kwargs dictionaries from a kwarg dictionary with values
    that are length-N lists, or possibly length-1 lists or scalars that should be broadcasted up to
    length-N lists.
    """
    if N is None:
        N = max(len(v) for v in dol.values() if hasattr(v, '__len__'))
    # Loop through broadcast range
    for i in range(N):
        out = {}
        for k, v in iteritems(dol):
            try:
                out[k] = v[i]
            except IndexError:  # It's list-like, but too short.
                if len(v) != 1:
                    raise GalSimIncompatibleValuesError(
                        "Cannot broadcast kwargs of different non-length-1 lengths.", dol=dol)
                out[k] = v[0]
            except TypeError:  # Value is not list-like, so broadcast it in its entirety.
                out[k] = v
            except KeyboardInterrupt:
                raise
            except: # pragma: no cover
                raise GalSimIncompatibleValuesError(
                    "Cannot broadcast kwarg {0}={1}".format(k, v), dol=dol)
        yield out

def structure_function(image):
    """Estimate the angularly-averaged structure function of a 2D random field.

    The angularly-averaged structure function D(r) of the 2D field phi is defined as:

    D(|r|) = <|phi(x) - phi(x+r)|^2>

    where the x and r on the RHS are 2D vectors, but the |r| on the LHS is just a scalar length.

    The image must have its `scale` attribute defined.  It will be used in the calculations to
    set the scale of the radial distances.

    @param image  Image containing random field realization.

    @returns      A python callable mapping a separation length r to the estimate of the structure
                  function D(r).
    """
    from .table import LookupTable2D
    array = image.array
    nx, ny = array.shape
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

def combine_wave_list(*args):
    """Combine wave_list attributes of all objects in obj_list while respecting blue_limit and
    red_limit attributes.  Should work with SEDs, Bandpasses, and ChromaticObjects.

    @param obj_list  List of SED, Bandpass, or ChromaticObject objects.
    @returns        wave_list, blue_limit, red_limit
    """
    from .sed import SED
    from .bandpass import Bandpass
    from .gsobject import GSObject
    from .chromatic import ChromaticObject
    if len(args) == 1:
        if isinstance(args[0],
                      (SED, Bandpass, ChromaticObject, GSObject)):
            args = [args[0]]
        elif isinstance(args[0], (list, tuple)):
            args = args[0]
        else:
            raise TypeError("Single input argument must be an SED, Bandpass, GSObject, "
                            "ChromaticObject or a (possibly mixed) list of them.")

    blue_limit = 0.0
    red_limit = np.inf
    wave_list = np.array([], dtype=float)
    for obj in args:
        if hasattr(obj, 'blue_limit'):
            blue_limit = max(blue_limit, obj.blue_limit)
        if hasattr(obj, 'red_limit'):
            red_limit = min(red_limit, obj.red_limit)
        wave_list = np.union1d(wave_list, obj.wave_list)
    wave_list = wave_list[(wave_list >= blue_limit) & (wave_list <= red_limit)]
    if blue_limit > red_limit:
        raise GalSimError("Empty wave_list intersection.")
    # Make sure both limits are included.
    if len(wave_list) > 0 and (wave_list[0] != blue_limit or wave_list[-1] != red_limit):
        wave_list = np.union1d([blue_limit, red_limit], wave_list)
    return wave_list, blue_limit, red_limit

def functionize(f):
    """ Decorate a function `f` which accepts scalar positional or keyword arguments, to accept
    arguments that can be either scalars or _functions_.  If the arguments include univariate
    (N-variate) functions, then the output will be a univariate (N-variate) function.  While it's
    okay to mix scalar and N-variate function arguments, it is an error to mix N-variate and
    M-variate function arguments.

    As an example:

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

    We can use arguments that accept keyword arguments too:

    >>> def f2(u, v=None):
    ...    if v is None:
    ...        v = 6.0
    ...    return u / v
    >>> result = decorated(2, f2)
    >>> print(result(12))  # 2 + (12./6) = 4.0
    >>> print(result(12, v=4))  # 2 + (12/4) = 5

    Note that you can also use python's decorator syntax:

    >>> @functionize
    >>> def f(x, y):
    ...     return x + y

    @param f  The function to be decorated.
    @returns  The decorated function.

    """
    @functools.wraps(f)
    def ff(*args, **kwargs):
        # First check if any of the arguments are callable...
        if not any(hasattr(arg, '__call__') for arg in args+tuple(kwargs.values())):
            return f(*args, **kwargs)  # ... if not, then keep output type a scalar ...
        else:
            def fff(*inner_args, **inner_kwargs): # ...else the output type is a function: `fff`.
                new_args = [arg
                            if not hasattr(arg, '__call__')
                            else arg(*inner_args, **inner_kwargs)
                            for arg in args]
                new_kwargs = dict([(k, v)
                                   if not hasattr(v, '__call__')
                                   else (k, v(*inner_args, **inner_kwargs))
                                   for k, v in iteritems(kwargs)])
                return f(*new_args, **new_kwargs)
            return fff
    return ff

def math_eval(str, other_modules=()):
    """Evaluate a string that may include numpy, np, or math commands.

    @param str              The string to evaluate
    @param other_modules    Other modules in addition to numpy, np, math to import as well.
                            Should be given as a list of strings.  [default: None]

    @returns Whatever the string evaluates to.
    """
    # Python 2 and 3 have a different syntax for exec with globals() dict.
    # The exec_ function lets us use the Python 3 syntax even in Python 2.
    from future.utils import exec_
    gdict = globals().copy()
    exec_('import galsim', gdict)
    exec_('import numpy', gdict)
    exec_('import numpy as np', gdict)
    exec_('import math', gdict)
    exec_('import coord', gdict)
    for m in other_modules:  # pragma: no cover  (We don't use this.)
        exec_('import ' + m, gdict)
    return eval(str, gdict)

def binomial(a, b, n):
    """Return xy coefficients of (ax + by)^n ordered by descending powers of a.

    For example:

    # (x + y)^3 = 1 x^3 + 3 x^2 y + 3 x y^2 + 1 y^3
    >>>  print(binomial(1, 1, 3))
    array([ 1.,  3.,  3.,  1.])


    # (2 x + y)^3 = 8 x^3 + 12 x^2 y + 6 x y^2 + 1 y^3
    >>>  print(binomial(2, 1, 3))
    array([ 8.,  12.,  6.,  1.])

    @param a    First scalar in binomial to be expanded.
    @param b    Second scalar in binomial to be expanded.
    @param n    Exponent of expansion.
    @returns    Array of coefficients in expansion.
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

    @param image    Image from which to compute moments
    @param origin   Optional origin in image coordinates used to compute Mx and My
                    [default: galsim.PositionD(0, 0)].
    @returns  Dict with entries for [M0, Mx, My, Mxx, Myy, Mxy]
    """
    from .position import PositionD
    if origin is None:
        origin = PositionD(0,0)
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

    @param arg   Either a galsim.Image or the output of unweighted_moments(image).
    @returns  Dict with entries for [rsqr, e1, e2]
    """
    from .image import Image
    if isinstance(arg, Image):
        arg = unweighted_moments(arg)
    rsqr = arg['Mxx'] + arg['Myy']
    return dict(rsqr=rsqr, e1=(arg['Mxx']-arg['Myy'])/rsqr, e2=2*arg['Mxy']/rsqr)

def rand_with_replacement(n, n_choices, rng, weight=None, _n_rng_calls=False):
    """Select some number of random choices from a list, with replacement, using a supplied RNG.

    `n` random choices with replacement are made assuming that those choices should range from 0 to
    `n_choices`-1, so they can be used as indices in a list/array.  If `weight` is supplied, then
    it should be an array of length `n_choices` that ranges from 0-1, and can be used to make
    weighted choices from the list.

    @param n           Number of random selections to make.
    @param n_choices   Number of entries from which to choose.
    @param rng         RNG to use.  Should be a galsim.BaseDeviate.
    @param weight      Optional list of weight factors to use for weighting the selection of
                       random indices.
    @returns a NumPy array of length `n` containing the integer-valued indices that were selected.
    """
    from .random import BaseDeviate, UniformDeviate
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
    """Find SED or Bandpass file, possibly adding share_dir/subdir.
    """
    from . import meta_data
    import os

    if os.path.isfile(filename):
        return True, filename

    new_filename = os.path.join(meta_data.share_dir, subdir, filename)
    if os.path.isfile(new_filename):
        return True, new_filename
    else:
        return False, ''


# http://stackoverflow.com/a/6849299
class lazy_property(object):
    """
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value

# cf. Docstring Inheritance Decorator at:
# https://github.com/ActiveState/code/wiki/Python_index_1
# Although I modified it slightly, since the original recipe there had a bug that made it
# not work properly with 2 levels of sub-classing (e.g. Pixel <- Box <- GSObject)
class doc_inherit(object):
    """
    Docstring inheriting method descriptor
    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        for parent in cls.__bases__: # pragma: no branch
            parfunc = getattr(parent, self.name, None)
            if parfunc and getattr(parfunc, '__doc__', None): # pragma: no branch
                break

        if obj:
            return self.get_with_inst(obj, cls, parfunc)
        else:
            return self.get_no_inst(cls, parfunc)

    def get_with_inst(self, obj, cls, parfunc):
        @functools.wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)
        return self.use_parent_doc(f, parfunc)

    def get_no_inst(self, cls, parfunc):
        @functools.wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs): # pragma: no cover (without inst, this is not normally called.)
            return self.mthd(*args, **kwargs)
        return self.use_parent_doc(f, parfunc)

    def use_parent_doc(self, func, source):
        if source is None: # pragma: no cover
            raise NameError("Can't find '%s' in parents"%self.name)
        func.__doc__ = source.__doc__
        return func

# Assign an arbitrary ordering to weakref.ref so that it can be part of a heap.
class OrderedWeakRef(weakref.ref):
    def __lt__(self, other):
        return id(self) < id(other)


def nCr(n, r):
    """Combinations.  I.e., the number of ways to choose `r` distiguishable things from `n`
    distinguishable things.
    """
    from math import factorial
    if 0 <= r <= n:
        return factorial(n) // (factorial(r) * factorial(n-r))
    else:
        return 0

# From http://code.activestate.com/recipes/81253-weakmethod/
class WeakMethod(object):
    def __init__(self, f):
        self.f = f.__func__
        self.c = weakref.ref(f.__self__)
    def __call__(self, *args):
        if self.c() is None :  # pragma: no cover
            raise TypeError('Method called on dead object')
        return self.f(self.c(), *args)

def ensure_dir(target):
    """
    Make sure the directory for the target location exists, watching for a race condition

    In particular check if the OS reported that the directory already exists when running
    makedirs, which can happen if another process creates it before this one can
    """

    _ERR_FILE_EXISTS=17
    dir = os.path.dirname(target)
    if dir == '': return

    exists = os.path.exists(dir)
    if not exists:
        try:
            os.makedirs(dir)
        except OSError as err:  # pragma: no cover
            # check if the file now exists, which can happen if some other
            # process created the directory between the os.path.exists call
            # above and the time of the makedirs attempt.  This is OK
            if err.errno != _ERR_FILE_EXISTS:
                raise err

    elif exists and not os.path.isdir(dir):  # pragma: no cover
        raise OSError("tried to make directory '%s' "
                      "but a non-directory file of that "
                      "name already exists" % dir)


def find_out_of_bounds_position(x, y, bounds, grid=False):
    """Given arrays of x and y values that are known to contain at least one
    position that is out-of-bounds of the given bounds instance, return one
    such PositionD.

    @param x  Array of x values
    @param y  Array of y values
    @param bounds  Bounds instance
    @param grid  Bool indicating whether to check the outer product of x and y
                 (grid=True), or each sequential pair of x and y (grid=False).
                 If the latter, then x and y should have the same shape.

    @returns a PositionD from x and y that is out-of-bounds of bounds.
    """
    from .position import PositionD
    if grid:
        # It's enough to check corners for grid input
        for x_ in (np.min(x), np.max(x)):
            for y_ in (np.min(y), np.max(y)):
                if (x_ < bounds.xmin or x_ > bounds.xmax or
                    y_ < bounds.ymin or y_ > bounds.ymax):
                    return PositionD(x_, y_)
    else:
        # Faster to check all points than to iterate through them one-by-one?
        w = np.where((x < bounds.xmin) | (x > bounds.xmax) |
                     (y < bounds.ymin) | (y > bounds.ymax))
        if len(w[0]) > 0:
            return PositionD(x[w[0][0]], y[w[0][0]])
    raise GalSimError("No out-of-bounds position")
