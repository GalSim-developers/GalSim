# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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
import numbers

from . import _galsim
from .utilities import lazy_property, convert_interpolant, find_out_of_bounds_position
from .utilities import basestring
from .bounds import BoundsD
from .errors import GalSimRangeError, GalSimBoundsError, GalSimValueError
from .errors import GalSimIncompatibleValuesError, galsim_warn
from .errors import GalSimNotImplementedError
from .interpolant import Interpolant

def _str_array(a):
    # Used by both LookupTable.__str__ and LookupTable2D.__str__
    # to write the x, f, etc. numpy arrays in an abbreviated form so they don't fill the
    # screen with numbers.
    # 1. Write the whole array if it has at most 5 values,
    # 2. Just write the first two and last two values in the array if longer.
    # 3. linewidth defaults to 75, which adds annoying linebreaks here.
    #    1000 should be big enough to mean "never".
    with np.printoptions(threshold=5, edgeitems=2, linewidth=1000):
        return repr(a)

class LookupTable(object):
    """
    LookupTable represents a lookup table to store function values that may be slow to calculate,
    for which interpolating from a lookup table is sufficiently accurate.

    A LookupTable may be constructed from two arrays (lists, tuples, or NumPy arrays of
    floats/doubles)::

        >>> args = [...]
        >>> vals = []
        >>> for arg in args:
        ...     val = calculateVal(arg)
        ...     vals.append(val)
        >>> table = galsim.LookupTable(x=args,f=vals)

    Then you can use this table as a replacement for the slow calculation::

        >>> other_args = [...]
        >>> for arg in other_args:
        ...     val = table(arg)
        ...     [... use val ...]


    The default interpolation method is a natural cubic spline.  This is usually the best choice,
    but we also provide other options, which can be specified by the ``interpolant`` kwarg.  The
    choices include 'floor', 'ceil', 'linear', 'spline', or a `galsim.Interpolant` object:

    - 'floor' takes the value from the previous argument in the table.
    - 'ceil' takes the value from the next argument in the table.
    - 'nearest' takes the value from the nearest argument in the table.
    - 'linear' does linear interpolation between these two values.
    - 'spline' uses a cubic spline interpolation, so the interpolated values are smooth at
      each argument in the table.
    - a `galsim.Interpolant` object or a string convertible to one.  This option can be used for
      `Lanczos` or `Quintic` interpolation, for example.

    Note that specifying the string 'nearest' or 'linear' will use a LookupTable-optimized
    interpolant instead of `galsim.Nearest` or `galsim.Linear`, though the latter options can still
    be used by passing an `Interpolant` object instead of a string.  Also note that to use a
    `galsim.Interpolant` in a LookupTable, the input data must be equally spaced, or logarithmically
    spaced if ``x_log`` is set to True (see below).  Finally, although natural cubic spline used
    when interpolant='spline' and the cubic convolution interpolant used when the interpolant
    is `galsim.Cubic` both produce piecewise cubic polynomial interpolations, their treatments of
    the continuity of derivatives are different (the natural spline is smoother).

    There are also two factory functions which can be used to build a LookupTable:

        `LookupTable.from_func`
                makes a LookupTable from a callable function

        `LookupTable.from_file`
                reads in a file of x and f values.

    The user can also opt to interpolate in log(x) and/or log(f) (if not using a
    `galsim.Interpolant`), though this is not the default.  It may be a wise choice depending on the
    particular function, e.g., for a nearly power-law f(x) (or at least one that is locally
    power-law-ish for much of the x range) then it might be a good idea to interpolate in log(x) and
    log(f) rather than x and f.

    Parameters:
        x:              The list, tuple, or NumPy array of ``x`` values.
        f:              The list, tuple, or NumPy array of ``f(x)`` values.
        interpolant:    Type of interpolation to use, with the options being 'floor', 'ceil',
                        'nearest', 'linear', 'spline', or a `galsim.Interpolant` or string
                        convertible to one.  [default: 'spline']
        x_log:          Set to True if you wish to interpolate using log(x) rather than x.  Note
                        that all inputs / outputs will still be x, it's just a question of how the
                        interpolation is done. [default: False]
        f_log:          Set to True if you wish to interpolate using log(f) rather than f.  Note
                        that all inputs / outputs will still be f, it's just a question of how the
                        interpolation is done. [default: False]
    """
    def __init__(self, x, f, interpolant='spline', x_log=False, f_log=False):
        self.x_log = x_log
        self.f_log = f_log

        # Check if interpolant is a string that we understand.  If not, try convert_interpolant
        if interpolant in ('nearest', 'linear', 'ceil', 'floor', 'spline'):
            self._interp1d = None
        else:
            self._interp1d = convert_interpolant(interpolant)
        self.interpolant = interpolant

        # Sanity checks
        if len(x) != len(f):
            raise GalSimIncompatibleValuesError("Input array lengths don't match", x=x, f=f)
        if len(x) < 2:
            raise GalSimValueError("Input arrays too small to interpolate", x)

        # turn x and f into numpy arrays so that all subsequent math is possible (unlike for
        # lists, tuples).  Also make sure the dtype is float
        x = np.asarray(x, dtype=float)
        if np.all(x[1:] >= x[:-1]):
            # Already sorted (a common case, so avoid the sort.
            self.x = np.ascontiguousarray(x, dtype=float)
            self.f = np.ascontiguousarray(f, dtype=float)
        else:
            f = np.asarray(f, dtype=float)
            s = np.argsort(x)
            self.x = np.ascontiguousarray(x[s])
            self.f = np.ascontiguousarray(f[s])

        self._x_min = self.x[0]
        self._x_max = self.x[-1]
        if self._x_min == self._x_max:
            raise GalSimValueError("All x values are equal", x)
        if self.x_log and self.x[0] <= 0.:
            raise GalSimValueError("Cannot interpolate in log(x) when table contains x<=0.", x)
        if self.f_log and np.any(self.f <= 0.):
            raise GalSimValueError("Cannot interpolate in log(f) when table contains f<=0.", f)

        # Check equal-spaced arrays
        if self._interp1d is not None:
            if self.x_log:
                ratio = self.x[1:]/self.x[:-1]
                if not np.allclose(ratio, ratio[0]):
                    raise GalSimIncompatibleValuesError(
                        "Cannot use a galsim.Interpolant with x_log=True unless log(x) is "
                        "equally spaced.",
                        interpolant=interpolant, x_log=x_log, x=x)
            else:
                dx = np.diff(self.x)
                if not np.allclose(dx, dx[0]):
                    raise GalSimIncompatibleValuesError(
                        "Cannot use a galsim.Interpolant with x_log=False unless x is "
                        "equally spaced.",
                        interpolant=interpolant, x_log=x_log, x=x)

    @lazy_property
    def _tab(self):
        # Store these as attributes, so don't need to worry about C++ layer persisting them.
        self._x = np.log(self.x) if self.x_log else self.x
        self._f = np.log(self.f) if self.f_log else self.f

        _x = self._x.__array_interface__['data'][0]
        _f = self._f.__array_interface__['data'][0]
        if self._interp1d is not None:
            return _galsim._LookupTable(_x, _f, len(self._x), self._interp1d._i)
        else:
            return _galsim._LookupTable(_x, _f, len(self._x), self.interpolant)

    @property
    def x_min(self):
        """The minimum x value in the lookup table.
        """
        return self._x_min

    @property
    def x_max(self):
        """The maximum x value in the lookup table.
        """
        return self._x_max

    def __len__(self): return len(self.x)

    def __call__(self, x):
        """Interpolate the `LookupTable` to get ``f(x)`` at some ``x`` value(s).

        When the `LookupTable` object is called with a single argument, it returns the value at that
        argument.  An exception will be thrown automatically if the ``x`` value is outside the
        range of the original tabulated values.  The value that is returned is the same type as
        that provided as an argument, e.g., if a single value ``x`` is provided then a single value
        of ``f`` is returned; if a tuple of ``x`` values is provided then a tuple of ``f`` values
        is returned; and so on.  Even if interpolation was done using the ``x_log`` option, the
        user should still provide ``x`` rather than ``log(x)``.

        Parameters:
            x:      The ``x`` value(s) for which ``f(x)`` should be calculated via interpolation on
                    the original ``(x,f)`` lookup table.  ``x`` can be a single float/double, or a
                    tuple, list, or arbitrarily shaped 1- or 2-dimensional NumPy array.

        Returns:
            the interpolated ``f(x)`` value(s).
        """
        orig_x = x
        # Handle the log(x) if necessary
        if self.x_log:
            x = np.log(x)

        x = np.asarray(x, dtype=float)
        try:
            if x.shape == ():
                f = self._tab.interp(float(x))
            else:
                dimen = len(x.shape)
                if dimen > 1:
                    f = np.empty_like(x.ravel(), dtype=float)
                    xx = x.astype(float,copy=False).ravel()
                    _xx = xx.__array_interface__['data'][0]
                    _f = f.__array_interface__['data'][0]
                    self._tab.interpMany(_xx, _f, len(xx))
                    f = f.reshape(x.shape)
                else:
                    f = np.empty_like(x, dtype=float)
                    xx = x.astype(float,copy=False)
                    _xx = xx.__array_interface__['data'][0]
                    _f = f.__array_interface__['data'][0]
                    self._tab.interpMany(_xx, _f, len(xx))
        except RuntimeError:
            # If there were points outside the valid range, this will have raised an exception.
            # so call _check_range to give a better error message.
            self._check_range(orig_x)
            raise  # pragma: no cover (shouldn't be able to reach here, but just in case.)

        # Handle the log(f) if necessary
        if self.f_log:
            f = np.exp(f)
        return f

    def integrate(self, x_min=None, x_max=None):
        r"""Calculate an estimate of the integral of the tabulated function from x_min to x_max:

        .. math::

            \int_{x_\mathrm{min}}^{x_\mathrm{max}} f(x) dx

        This function is not implemented for LookupTables that use log for either x or f,
        or that use a ``galsim.Interpolant``.  Also, if x_min or x_max are beyond the range
        of the tabulated function, the function will be considered to be zero there.

        .. note::

            The simplest version of this function is equivalent in functionality to the numpy
            ``trapz`` function.  However, it is usually significantly faster.  If you have a
            time-critical integration for which you are currently using ``np.trapz``::

                >>> ans = np.trapz(f, x)

            the following replacement may be faster::

                >>> ans = galsim.trapz(f, x)

            which is an alias for::

                >>> ans = galsim._LookupTable(x, f, 'linear').integrate()

        Parameters:
            x_min:      The minimum abscissa to use for the integral.  [default: None, which
                        means to use self.x_min]
            x_max:      The maximum abscissa to use for the integral.  [default: None, which
                        means to use self.x_max]

        Returns:
            an estimate of the integral
        """
        if self.x_log:
            raise GalSimNotImplementedError("log x spacing not implemented yet.")
        if self.f_log:
            raise GalSimNotImplementedError("log f values not implemented yet.")
        if not isinstance(self.interpolant, basestring):
            raise GalSimNotImplementedError(
                "Integration with interpolant=%s is not implemented."%(self.interpolant))
        if x_min is None:
            x_min = self.x_min
        else:
            x_min = max(x_min, self.x_min)
        if x_max is None:
            x_max = self.x_max
        else:
            x_max = min(x_max, self.x_max)

        if x_min < x_max:
            return self._tab.integrate(x_min, x_max)
        elif x_min == x_max:
            return 0.
        else:
            return -self.integrate(x_max, x_min)

    def integrate_product(self, g, x_min=None, x_max=None, x_factor=1.):
        r"""Calculate an estimate of the integral of the tabulated function multiplied by a second
        function from x_min to x_max:

        .. math::

            \int_{x_\mathrm{min}}^{x_\mathrm{max}} f(x) g(x) dx

        If the second function, :math:`g(x)`, is another `LookupTable`, then the quadrature will
        use the abscissae from both that function and :math:`f(x)` (i.e. ``self``).
        Otherwise, the second function will be evaluated at the abscissae of :math:`f(x)`.

        This function is not implemented for LookupTables that use log for either x or f,
        or that use a ``galsim.Interpolant``.  Also, if x_min or x_max are beyond the range
        of either tabulated function, the function will be considered to be zero there.

        Also, the second function :math:`g(x)` is always approximated with linear interpolation
        between the abscissae, even if it is a `LookupTable` with a different specified
        interpolation.

        Parameters:
            g:          The function to multiply by the current function for the integral.
            x_min:      The minimum abscissa to use for the integral.  [default: None, which
                        means to use self.x_min]
            x_max:      The maximum abscissa to use for the integral.  [default: None, which
                        means to use self.x_max]
            x_factor:   Optionally scale the x values of f by this factor when doing the integral.
                        I.e. Find :math:`\int f(x x_\mathrm{factor}) g(x) dx`. [default: 1]

        Returns:
            an estimate of the integral
        """
        if self.x_log:
            raise GalSimNotImplementedError("log x spacing not implemented yet.")
        if self.f_log:
            raise GalSimNotImplementedError("log f values not implemented yet.")
        if not isinstance(self.interpolant, basestring):
            raise GalSimNotImplementedError(
                "Integration with interpolant=%s is not implemented."%(self.interpolant))
        if x_min is None:
            x_min = self.x_min / x_factor
        else:
            x_min = max(x_min, self.x_min / x_factor)
        if x_max is None:
            x_max = self.x_max / x_factor
        else:
            x_max = min(x_max, self.x_max / x_factor)
        if x_min > x_max:
            return -self.integrate_product(g, x_max, x_min, x_factor)
        elif x_min == x_max:
            return 0.

        if isinstance(g, LookupTable):
            x_min = max(x_min, g.x_min)
            x_max = min(x_max, g.x_max)
            if x_min >= x_max:
                return 0.
        else:
            gx = self.x / x_factor
            gx = gx[(gx >= x_min) & (gx <= x_max)]
            gx = np.union1d(gx, [x_min, x_max])
            # Let this raise an appropriate error if g is not a valid function over this domain.
            gf = g(gx)
            # If g is a constant function (like lambda wave: 1), then this doesn't return
            # an array.  Make it one.
            try:
                len(gf)
            except TypeError:
                gf1 = gf
                gf = np.empty_like(gx, dtype=float)
                gf.fill(gf1)
            g = _LookupTable(gx, gf, 'linear')

        return self._tab.integrate_product(g._tab, float(x_min), float(x_max), float(x_factor))

    def _check_range(self, x):
        slop = (self.x_max - self.x_min) * 1.e-6
        if np.min(x,initial=self.x_min) < self.x_min - slop:
            raise GalSimRangeError("x value(s) below the range of the LookupTable.",
                                   x, self.x_min, self.x_max)
        if np.max(x,initial=self.x_max) > self.x_max + slop:  # pragma: no branch
            raise GalSimRangeError("x value(s) above the range of the LookupTable.",
                                   x, self.x_min, self.x_max)

    def getArgs(self):
        return self.x

    def getVals(self):
        return self.f

    def getInterp(self):
        return self.interpolant

    def isLogX(self):
        return self.x_log

    def isLogF(self):
        return self.f_log

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, LookupTable) and
                 np.array_equal(self.x,other.x) and
                 np.array_equal(self.f,other.f) and
                 self.x_log == other.x_log and
                 self.f_log == other.f_log and
                 self.interpolant == other.interpolant))
    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self):
        # Cache this in case self.x, self.f are long.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.LookupTable", tuple(self.x), tuple(self.f), self.x_log,
                               self.f_log, self.interpolant))
        return self._hash


    def __repr__(self):
        return 'galsim.LookupTable(x=array(%r), f=array(%r), interpolant=%r, x_log=%r, f_log=%r)'%(
            self.x.tolist(), self.f.tolist(), self.interpolant, self.x_log, self.f_log)

    def __str__(self):
        s = 'galsim.LookupTable(x=%s, f=%s'%(_str_array(self.x), _str_array(self.f))
        if self.interpolant != 'spline':
            s += ', interpolant=%r'%(self.interpolant)
        if self.x_log:
            s += ', x_log=True'
        if self.f_log:
            s += ', f_log=True'
        s += ')'
        return s

    @classmethod
    def from_file(cls, file_name, interpolant='spline', x_log=False, f_log=False, amplitude=1.0):
        """Create a `LookupTable` from a file of x, f values.

        This reads in a file, which should contain two columns with the x and f values.

        Parameters:
            file_name:      A file from which to read the ``(x,f)`` pairs.
            interpolant:    Type of interpolation to use. [default: 'spline']
            x_log:          Whether the x values should be uniform in log rather than lienar.
                            [default: False]
            f_log:          Whether the f values should be interpolated using their logarithms
                            rather than their raw values. [default: False]
            amplitude:      An optional scaling of the f values relative to the values in the file
                            [default: 1.0]
        """
        # We don't require pandas as a dependency, but if it's available, this is much faster.
        # cf. http://stackoverflow.com/questions/15096269/the-fastest-way-to-read-input-in-python
        ParserError = AttributeError # In case we don't get to the line below where we import
                                     # it from pandas.
        try:
            import pandas
            from pandas.errors import ParserError
            data = pandas.read_csv(file_name, comment='#', delim_whitespace=True, header=None)
            data = data.values.transpose()
        except (ImportError, AttributeError, ParserError):
            data = np.loadtxt(file_name).transpose()
        if data.shape[0] != 2:
            raise GalSimValueError("File provided for LookupTable does not have 2 columns",
                                   file_name)
        x=data[0]
        f=data[1]
        if amplitude != 1.0:
            f[:] *= amplitude
        return LookupTable(x, f, interpolant=interpolant, x_log=x_log, f_log=f_log)

    @classmethod
    def from_func(cls, func, x_min, x_max, npoints=2000, interpolant='spline',
                  x_log=False, f_log=False):
        """Create a `LookupTable` from a callable function

        This constructs a `LookupTable` over the given range from x_min and x_max, calculating the
        corresponding f values from the given function (technically any callable object).

        Parameters:
            func:           A callable function.
            x_min:          The minimum x value at which to evalue the function and store in the
                            lookup table.
            x_max:          The maximum x value at which to evalue the function and store in the
                            lookup table.
            npoints:        Number of x values at which to evaluate the function. [default: 2000]
            interpolant:    Type of interpolation to use. [default: 'spline']
            x_log:          Whether the x values should be uniform in log rather than lienar.
                            [default: False]
            f_log:          Whether the f values should be interpolated using their logarithms
                            rather than their raw values. [default: False]
        """
        if x_log:
            x = np.exp(np.linspace(np.log(x_min), np.log(x_max), npoints))
        else:
            x = np.linspace(x_min, x_max, npoints)
        f = np.array([func(xx) for xx in x], dtype=float)
        return cls(x, f, interpolant=interpolant, x_log=x_log, f_log=f_log)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_tab',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

def _LookupTable(x, f, interpolant='spline', x_log=False, f_log=False):
    """Make a `LookupTable` but without using any of the sanity checks or array manipulation used
    in the normal initializer.

    The input x values must be already sorted.

    Parameters:
        x:              Strictly increasing NumPy array of ``x`` values.
        f:              NumPy array of ``f(x)`` values.
        interpolant:    Type of interpolation to use, with the options being 'floor', 'ceil',
                        'nearest', 'linear', 'spline', or a `galsim.Interpolant` or string
                        convertible to one.  [default: 'spline']
        x_log:          Set to True if you wish to interpolate using log(x) rather than x.  Note
                        that all inputs / outputs will still be x, it's just a question of how the
                        interpolation is done. [default: False]
        f_log:          Set to True if you wish to interpolate using log(f) rather than f.  Note
                        that all inputs / outputs will still be f, it's just a question of how the
                        interpolation is done. [default: False]
    """
    ret = LookupTable.__new__(LookupTable)
    ret.x = np.ascontiguousarray(x, dtype=float)
    ret.f = np.ascontiguousarray(f, dtype=float)
    ret.interpolant = interpolant
    ret.x_log = x_log
    ret.f_log = f_log
    ret._x_min = ret.x[0]
    ret._x_max = ret.x[-1]
    if interpolant in ('nearest', 'linear', 'ceil', 'floor', 'spline'):
        ret._interp1d = None
    else:
        ret._interp1d = convert_interpolant(interpolant)
    return ret


def trapz(f, x):
    """Integrate f(x) using the trapezoidal rule.

    Equivalent to np.trapz(f,x) for 1d array inputs.  Intended as a drop-in replacement,
    which is usually faster.

    Parameters:
        f:      The ordinates of the function to integrate.
        x:      The abscissae of the function to integrate.

    Returns:
        Estimate of the integral.
    """
    if len(x) >= 2:
        return _LookupTable(x,f,'linear').integrate()
    else:
        return 0.


class LookupTable2D(object):
    """
    LookupTable2D represents a 2-dimensional lookup table to store function values that may be slow
    to calculate, for which interpolating from a lookup table is sufficiently accurate.  A
    LookupTable2D is also useful for evaluating periodic 2-d functions given samples from a single
    period.

    A LookupTable2D representing the function f(x, y) may be constructed from a list or array of
    ``x`` values, a list or array of ``y`` values, and a 2D array of function evaluations at all
    combinations of x and y values.  For instance::

        >>> x = np.arange(5)
        >>> y = np.arange(8)
        >>> z = x + y[:, np.newaxis]  # function is x + y, dimensions of z are (8, 5)
        >>> tab2d = galsim.LookupTable2D(x, y, z)

    To evaluate new function values with the lookup table, use the () operator::

        >>> print tab2d(2.2, 3.3)
        5.5

    The () operator can also accept sequences (lists, tuples, numpy arrays, ...) for the x and y
    arguments at which to evaluate the LookupTable2D.  Normally, the x and y sequences should have
    the same length, which will also be the length of the output sequence::

        >>> print tab2d([1, 2], [3, 5])
        array([ 4., 7.])

    If you add ``grid=True`` as an additional kwarg, however, then the () operator will generate
    interpolated values at the outer product of x-values and y-values.  So in this case, the x and
    y sequences can have different lengths Nx and Ny, and the result will be a 2D array with
    dimensions (Nx, Ny)::

        >>> print tab2d([1, 2], [3, 5], grid=True)
        array([[ 4., 6.],
               [ 5., 7.]])

    The default interpolation method is linear.  Other choices for the interpolant are:

        - 'floor'
        - 'ceil'
        - 'nearest'
        - 'spline' (a Catmull-Rom cubic spline).
        - a `galsim.Interpolant` or string convertible to one.

    ::

        >>> tab2d = galsim.LookupTable2D(x, y, z, interpolant='floor')
        >>> tab2d(2.2, 3.7)
        5.0
        >>> tab2d = galsim.LookupTable2D(x, y, z, interpolant='ceil')
        >>> tab2d(2.2, 3.7)
        7.0
        >>> tab2d = galsim.LookupTable2D(x, y, z, interpolant='nearest')
        >>> tab2d(2.2, 3.7)
        6.0

    For interpolant='spline' or a `galsim.Interpolant`, the input arrays must be uniformly spaced.
    For interpolant='spline', the derivatives df / dx, df / dy, and d^2 f / dx dy at grid-points may
    also optionally be provided if they're known, which will generally yield a more accurate
    interpolation (these derivatives will be estimated from finite differences if they're not
    provided).

    The ``edge_mode`` keyword describes how to handle extrapolation beyond the initial input range.
    Possibilities include:

      - 'raise': raise an exception.  (This is the default.)
      - 'warn': issues a warning, then falls back to edge_mode='constant'.
      - 'constant': Return a constant specified by the ``constant`` keyword.
      - 'wrap': infinitely wrap the initial range in both directions.

    In order for LookupTable2D to determine the wrapping period when edge_mode='wrap', either the
    x and y grid points need to be equally spaced (in which case the x-period is inferred as
    len(x)*(x[1]-x[0]) and similarly for y), or the first/last row/column of f must be identical,
    in which case the x-period is inferred as x[-1] - x[0].  (If both conditions are satisfied
    (equally-spaced x and y and identical first/last row/column of f, then the x-period is inferred
    as len(x)*(x[1]-x[0]))::

        >>> x = np.arange(5)
        >>> y = np.arange(8)
        >>> z = x + y[:, np.newaxis]  # function is x + y, dimensions of z are (8, 5)
        >>> tab2d = galsim.LookupTable2D(x, y, z, edge_mode='raise')
        >>> tab2d(7, 7)
        ValueError: Extrapolating beyond input range.

        >>> tab2d = galsim.LookupTable2D(x, y, z, edge_mode='constant', constant=1.0)
        1.0

        >>> tab2d = galsim.LookupTable2D(x, y, z, edge_mode='wrap')
        ValueError: Cannot wrap `f` array with unequal first/last column/row.

    We extend the x and y arrays with a uniform spacing, though any monotonic spacing would work.
    Note that the [(0,1), (0,1)] argument in np.pad below extends the z array by 0 rows/columns in
    the leading direction, and 1 row/column in the trailing direction::

        >>> x = np.append(x, x[-1] + (x[-1]-x[-2]))
        >>> y = np.append(y, y[-1] + (y[-1]-y[-2]))
        >>> z = np.pad(z, [(0,1), (0,1)], mode='wrap')
        >>> tab2d = galsim.LookupTable2D(x, y, z, edge_mode='wrap')
        >>> tab2d(2., 2.)
        4.0
        >>> tab2d(2.+5, 2.)  # The period is 5 in the x direction
        4.0
        >>> tab2d(2.+3*5, 2.+4*8)  # The period is 8 in the y direction
        4.0

    Parameters:
        x:              Strictly increasing array of ``x`` positions at which to create table.
        y:              Strictly increasing array of ``y`` positions at which to create table.
        f:              Nx by Ny input array of function values.
        dfdx:           Optional first derivative of f wrt x.  Only used if interpolant='spline'.
                        [default: None]
        dfdy:           Optional first derivative of f wrt y.  Only used if interpolant='spline'.
                        [default: None]
        d2fdxdy:        Optional cross derivative of f wrt x and y.  Only used if
                        interpolant='spline'.  [default: None]
        interpolant:    Type of interpolation to use.  One of 'floor', 'ceil', 'nearest', 'linear',
                        'spline', or a `galsim.Interpolant` or string convertible to one.
                        [default: 'linear']
        edge_mode:      Keyword controlling how extrapolation beyond the input range is handled.
                        See above for details.  [default: 'raise']
        constant:       A constant to return when extrapolating beyond the input range and
                        ``edge_mode='constant'``.  [default: 0]
    """
    def __init__(self, x, y, f, dfdx=None, dfdy=None, d2fdxdy=None,
                 interpolant='linear', edge_mode='raise', constant=0):
        if edge_mode not in ('raise', 'warn', 'wrap', 'constant'):
            raise GalSimValueError("Unknown edge_mode.", edge_mode,
                                   ('raise', 'warn', 'wrap', 'constant'))

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        f = np.asarray(f, dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)
        equal_spaced = np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])

        if not all(dx > 0):
            raise GalSimValueError("x input grids is not strictly increasing.", x)
        if not all(dy > 0):
            raise GalSimValueError("y input grids is not strictly increasing.", y)

        fshape = f.shape
        if fshape != (len(y), len(x)):
            raise GalSimIncompatibleValuesError(
                "Shape of f incompatible with lengths of x,y", f=f, x=x, y=y)

        # Check if interpolant is a string that we understand.  If not, try convert_interpolant
        if interpolant in ('nearest', 'linear', 'ceil', 'floor', 'spline'):
            self._interp2d = None
            padrange = 2 if interpolant == 'spline' else 1
        else:
            self._interp2d = convert_interpolant(interpolant)
            padrange = int(np.ceil(self._interp2d.xrange))
        self.interpolant = interpolant

        # Check if need equal-spaced arrays
        if (self._interp2d is not None or interpolant  == 'spline'):
            if not equal_spaced:
                raise GalSimIncompatibleValuesError(
                "Cannot use a galsim.Interpolant in LookupTable2D unless x and y are "
                "equally spaced.", interpolant=interpolant, x=x, y=y)

        self.edge_mode = edge_mode
        self.constant = float(constant)

        if self.edge_mode == 'wrap':
            # Can wrap if x and y arrays are equally spaced ...
            if equal_spaced:
                if 2*padrange > len(x) or 2*padrange > len(y):
                    raise GalSimValueError(
                        "Cannot wrap an image which is smaller than the Interpolant",
                        (x, y, interpolant))
                # Underlying Table2D requires us to extend x, y, and f.
                self.xperiod = x[-1]-x[0]+dx[0]
                self.yperiod = y[-1]-y[0]+dy[0]
                self.x0 = x[0]
                self.y0 = y[0]
                x = np.hstack([x[0]-np.cumsum([dx[0]]*(padrange-1)),
                               x,
                               x[-1]+np.cumsum([dx[0]]*padrange)])
                y = np.hstack([y[0]-np.cumsum([dy[0]]*(padrange-1)),
                               y,
                               y[-1]+np.cumsum([dy[0]]*padrange)])
                f = np.pad(f, [(padrange-1, padrange)]*2, mode='wrap')
            # Can also wrap non-uniform grids if edges match
            elif (all(f[0] == f[-1]) and all(f[:,0] == f[:,-1])):
                self.x0 = x[0]
                self.y0 = y[0]
                self.xperiod = x[-1] - x[0]
                self.yperiod = y[-1] - y[0]
            else:
                raise GalSimIncompatibleValuesError(
                    "Cannot use edge_mode='wrap' unless either x and y are equally "
                    "spaced or first/last row/column of f are identical.",
                    edge_mode=edge_mode, x=x, y=y, f=f)

        self.x = np.ascontiguousarray(x)
        self.y = np.ascontiguousarray(y)
        self.f = np.ascontiguousarray(f)

        der_exist = [kw is not None for kw in [dfdx, dfdy, d2fdxdy]]
        if self.interpolant == 'spline':
            if any(der_exist):
                if not all(der_exist):
                    raise GalSimIncompatibleValuesError(
                        "Must specify all of dfdx, dfdy, d2fdxdy if one is specified",
                        dfdx=dfdx, dfdy=dfdy, d2fdxdy=d2fdxdy)
            else:
                # Use finite differences if derivatives not provided
                dfdx = np.empty_like(f)
                diffx = self.x[2:] - self.x[:-2]
                dfdx[:, 1:-1] = (f[:, 2:] - f[:, :-2])/diffx
                dfdx[:, 0] = (f[:, 1] - f[:, 0])/dx[0]
                dfdx[:, -1] = (f[:, -1] - f[:, -2])/dx[-1]

                dfdy = np.empty_like(f)
                diffy = self.y[2:] - self.y[:-2]
                dfdy[1:-1, :] = (f[2:, :] - f[:-2, :])/diffy[:,None]
                dfdy[0, :] = (f[1, :] - f[0, :])/dy[0]
                dfdy[-1, :] = (f[-1, :] - f[-2, :])/dy[-1]

                d2fdxdy = np.empty_like(f)
                d2fdxdy[1:-1, :] = (dfdx[2:, :] - dfdx[:-2, :])/diffy[:,None]
                d2fdxdy[0, :] = (dfdx[1, :] - dfdx[0, :])/dy[0]
                d2fdxdy[-1, :] = (dfdx[-1, :] - dfdx[-2, :])/dy[-1]
        else:
            if any(der_exist):
                raise GalSimIncompatibleValuesError(
                    "Only specify dfdx, dfdy, d2fdxdy if interpolant is 'spline'.",
                    dfdx=dfdx, dfdy=dfdy, d2fdxdy=d2fdxdy, interpolant=interpolant)

        if dfdx is not None:
            dfdx = np.ascontiguousarray(dfdx, dtype=float)
            dfdy = np.ascontiguousarray(dfdy, dtype=float)
            d2fdxdy = np.ascontiguousarray(d2fdxdy, dtype=float)

            if dfdx.shape != f.shape or dfdy.shape != f.shape or d2fdxdy.shape != f.shape:
                raise GalSimIncompatibleValuesError(
                    "derivative shapes must match f shape",
                    dfdx=dfdx, dfdy=dfdy, d2fdxdy=d2fdxdy)

        self.dfdx = dfdx
        self.dfdy = dfdy
        self.d2fdxdy = d2fdxdy

    @lazy_property
    def _tab(self):
        _x = self.x.__array_interface__['data'][0]
        _y = self.y.__array_interface__['data'][0]
        _f = self.f.__array_interface__['data'][0]
        if self._interp2d is not None:
            return _galsim._LookupTable2D(_x, _y, _f, len(self.x), len(self.y),
                                          self._interp2d._i)
        elif self.interpolant == 'spline':
            _dfdx = self.dfdx.__array_interface__['data'][0]
            _dfdy = self.dfdy.__array_interface__['data'][0]
            _d2fdxdy = self.d2fdxdy.__array_interface__['data'][0]
            return _galsim._LookupTable2D(_x, _y, _f, len(self.x), len(self.y),
                                          _dfdx, _dfdy, _d2fdxdy)
        else:
            return _galsim._LookupTable2D(_x, _y, _f, len(self.x), len(self.y),
                                          self.interpolant)

    def getXArgs(self):
        return self.x

    def getYArgs(self):
        return self.y

    def getVals(self):
        return self.f

    def _inbounds(self, x, y):
        """Return whether or not *all* coords specified by x and y are in bounds of the original
        interpolated array."""
        # Only used if edge_mode != 'wrap', so original x/y arrays are unmodified.
        return (np.min(x) >= self.x[0] and np.max(x) <= self.x[-1] and
                np.min(y) >= self.y[0] and np.max(y) <= self.y[-1])

    def _wrap_args(self, x, y):
        """Wrap points back into the fundamental period."""
        # Original x and y may have been modified, so need to use x0 and xperiod attributes here.
        #x = (x-self.x0) % self.xperiod + self.x0
        #y = (y-self.y0) % self.yperiod + self.y0
        _x = x.__array_interface__['data'][0]
        _y = y.__array_interface__['data'][0]
        _galsim.WrapArrayToPeriod(_x, len(x), self.x0, self.xperiod)
        _galsim.WrapArrayToPeriod(_y, len(y), self.y0, self.yperiod)
        return x, y

    @property
    def _bounds(self):
        # Only meaningful if edge_mode is 'raise' or 'warn', in which case original x/y arrays are
        # unmodified.
        return BoundsD(self.x[0], self.x[-1], self.y[0], self.y[-1])

    def _call_inbounds(self, x, y, grid=False):
        _x = x.__array_interface__['data'][0]
        _y = y.__array_interface__['data'][0]
        if grid:
            f = np.empty((len(y), len(x)), dtype=float)
            _f = f.__array_interface__['data'][0]
            self._tab.interpGrid(_x, _y, _f, len(x), len(y))
            return f
        else:
            f = np.empty_like(x, dtype=float)
            _f = f.__array_interface__['data'][0]
            self._tab.interpMany(_x, _y, _f, len(x))
            return f

    def _call_constant(self, x, y, grid=False):
        x = np.array(x, dtype=float, copy=False)
        y = np.array(y, dtype=float, copy=False)
        if grid:
            f = np.empty((len(y), len(x)), dtype=float)
            # Fill in interpolated values first, then go back and fill in
            # constants
            _x = x.__array_interface__['data'][0]
            _y = y.__array_interface__['data'][0]
            _f = f.__array_interface__['data'][0]
            self._tab.interpGrid(_x, _y, _f, len(x), len(y))
            badx = (x < self.x[0]) | (x > self.x[-1])
            bady = (y < self.y[0]) | (y > self.y[-1])
            f[bady, :] = self.constant
            f[:, badx] = self.constant
            return f
        else:
            # Start with constant array, then interpolate good positions
            f = np.empty_like(x, dtype=float)
            f.fill(self.constant)
            good = ((x >= self.x[0]) & (x <= self.x[-1]) &
                    (y >= self.y[0]) & (y <= self.y[-1]))
            xx = np.ascontiguousarray(x[good].ravel(), dtype=float)
            yy = np.ascontiguousarray(y[good].ravel(), dtype=float)
            tmp = np.empty_like(xx, dtype=float)
            _xx = xx.__array_interface__['data'][0]
            _yy = yy.__array_interface__['data'][0]
            _tmp = tmp.__array_interface__['data'][0]
            self._tab.interpMany(_xx, _yy, _tmp, len(xx))
            f[good] = tmp
            return f

    def _call_raise(self, x, y, grid=False):
        if not self._inbounds(x, y):
            pos = find_out_of_bounds_position(x, y, self._bounds, grid)
            raise GalSimBoundsError("Extrapolating beyond input range.",
                                    pos, self._bounds)
        return self._call_inbounds(x, y, grid)

    def _call_warn(self, x, y, grid=False):
        if not self._inbounds(x, y):
            pos = find_out_of_bounds_position(x, y, self._bounds, grid)
            galsim_warn("Extrapolating beyond input range. {!r} not in {!r}".format(
                        pos, self._bounds))
        return self._call_constant(x, y, grid)

    def _call_wrap(self, x, y, grid=False):
        x, y = self._wrap_args(x, y)
        return self._call_inbounds(x, y, grid)

    def __call__(self, x, y, grid=False):
        """Interpolate at an arbitrary point or points.

        Parameters:
            x:      Either a single x value or an array of x values at which to interpolate.
            y:      Either a single y value or an array of y values at which to interpolate.
            grid:   Optional boolean indicating that output should be a 2D array corresponding
                    to the outer product of input values.  If False (default), then the output
                    array will be congruent to x and y.

        Returns:
            a scalar value if x and y are scalar, or a numpy array if x and y are arrays.
        """
        x1 = np.array(x, dtype=float, copy=self.edge_mode=='wrap')
        y1 = np.array(y, dtype=float, copy=self.edge_mode=='wrap')
        x2 = np.ascontiguousarray(x1.ravel(), dtype=float)
        y2 = np.ascontiguousarray(y1.ravel(), dtype=float)

        if self.edge_mode == 'raise':
            f = self._call_raise(x2, y2, grid)
        elif self.edge_mode == 'warn':
            f = self._call_warn(x2, y2, grid)
        elif self.edge_mode == 'wrap':
            f = self._call_wrap(x2, y2, grid)
        else: # constant
            f = self._call_constant(x2, y2, grid)

        if isinstance(x, numbers.Real):
            return f[0]
        else:
            if not grid:
                f = f.reshape(x1.shape)
            return f

    def _gradient_inbounds(self, x, y, grid=False):
        if grid:
            dfdx = np.empty((len(y), len(x)), dtype=float)
            dfdy = np.empty((len(y), len(x)), dtype=float)
            _x = x.__array_interface__['data'][0]
            _y = y.__array_interface__['data'][0]
            _dfdx = dfdx.__array_interface__['data'][0]
            _dfdy = dfdy.__array_interface__['data'][0]
            self._tab.gradientGrid(_x, _y, _dfdx, _dfdy, len(x), len(y))
            return dfdx, dfdy
        else:
            dfdx = np.empty_like(x)
            dfdy = np.empty_like(x)
            _x = x.__array_interface__['data'][0]
            _y = y.__array_interface__['data'][0]
            _dfdx = dfdx.__array_interface__['data'][0]
            _dfdy = dfdy.__array_interface__['data'][0]
            self._tab.gradientMany(_x, _y, _dfdx, _dfdy, len(x))
            return dfdx, dfdy

    def _gradient_raise(self, x, y, grid=False):
        if not self._inbounds(x, y):
            pos = find_out_of_bounds_position(x, y, self._bounds, grid)
            raise GalSimBoundsError("Extrapolating beyond input range.",
                                    pos, self._bounds)
        return self._gradient_inbounds(x, y, grid)

    def _gradient_warn(self, x, y, grid=False):
        if not self._inbounds(x, y):
            pos = find_out_of_bounds_position(x, y, self._bounds, grid)
            galsim_warn("Extrapolating beyond input range. {!r} not in {!r}".format(
                        pos, self._bounds))
        return self._gradient_constant(x, y, grid)

    def _gradient_wrap(self, x, y, grid=False):
        x, y = self._wrap_args(x, y)
        return self._gradient_inbounds(x, y, grid)

    def _gradient_constant(self, x, y, grid=False):
        x = np.array(x, dtype=float, copy=False)
        y = np.array(y, dtype=float, copy=False)
        if grid:
            dfdx = np.empty((len(y), len(x)), dtype=float)
            dfdy = np.empty((len(y), len(x)), dtype=float)
            _x = x.__array_interface__['data'][0]
            _y = y.__array_interface__['data'][0]
            _dfdx = dfdx.__array_interface__['data'][0]
            _dfdy = dfdy.__array_interface__['data'][0]
            self._tab.gradientGrid(_x, _y, _dfdx, _dfdy, len(x), len(y))
            badx = (x < self.x[0]) | (x > self.x[-1])
            bady = (y < self.y[0]) | (y > self.y[-1])
            dfdx[bady,:] = 0.0
            dfdx[:, badx] = 0.0
            dfdy[bady,:] = 0.0
            dfdy[:, badx] = 0.0
            return dfdx, dfdy
        else:
            dfdx = np.empty_like(x, dtype=float)
            dfdy = np.empty_like(x, dtype=float)
            dfdx.fill(0.0)
            dfdy.fill(0.0)
            good = ((x >= self.x[0]) & (x <= self.x[-1]) &
                    (y >= self.y[0]) & (y <= self.y[-1]))
            x = np.ascontiguousarray(x[good].ravel(), dtype=float)
            y = np.ascontiguousarray(y[good].ravel(), dtype=float)
            tmp1 = np.empty_like(x, dtype=float)
            tmp2 = np.empty_like(x, dtype=float)
            _x = x.__array_interface__['data'][0]
            _y = y.__array_interface__['data'][0]
            _tmp1 = tmp1.__array_interface__['data'][0]
            _tmp2 = tmp2.__array_interface__['data'][0]
            self._tab.gradientMany(_x, _y, _tmp1, _tmp2, len(x))
            dfdx[good] = tmp1
            dfdy[good] = tmp2
            return dfdx, dfdy

    def gradient(self, x, y, grid=False):
        """Calculate the gradient of the function at an arbitrary point or points.

        Parameters:
            x:      Either a single x value or an array of x values at which to compute
                    the gradient.
            y:      Either a single y value or an array of y values at which to compute
                    the gradient.
            grid:   Optional boolean indicating that output should be a 2-tuple of 2D arrays
                    corresponding to the outer product of input values.  If False (default),
                    then the output arrays will be congruent to x and y.

        Returns:
            A tuple of (dfdx, dfdy) where dfdx, dfdy are single values (if x,y were single
            values) or numpy arrays.
        """
        x1 = np.array(x, dtype=float, copy=self.edge_mode=='wrap')
        y1 = np.array(y, dtype=float, copy=self.edge_mode=='wrap')
        x2 = np.ascontiguousarray(x1.ravel(), dtype=float)
        y2 = np.ascontiguousarray(y1.ravel(), dtype=float)

        if self.edge_mode == 'raise':
            dfdx, dfdy = self._gradient_raise(x2, y2, grid)
        if self.edge_mode == 'warn':
            dfdx, dfdy = self._gradient_warn(x2, y2, grid)
        elif self.edge_mode == 'wrap':
            dfdx, dfdy = self._gradient_wrap(x2, y2, grid)
        else: # constant
            dfdx, dfdy = self._gradient_constant(x2, y2, grid)

        if isinstance(x, numbers.Real):
            return dfdx[0], dfdy[0]
        else:
            if not grid:
                dfdx = dfdx.reshape(x1.shape)
                dfdy = dfdy.reshape(x1.shape)
            return dfdx, dfdy

    def __str__(self):
        return "galsim.LookupTable2D(x=%s, y=%s, f=[%s,...,%s], interpolant=%r, edge_mode=%r)"%(
            _str_array(self.x), _str_array(self.y),
            _str_array(self.f[0]), _str_array(self.f[-1]),
            self.interpolant, self.edge_mode)

    def __repr__(self):
        return ("galsim.LookupTable2D(x=array(%r), y=array(%r), "
                "f=array(%r), interpolant=%r, edge_mode=%r, constant=%r)"%(
            self.x.tolist(), self.y.tolist(), self.f.tolist(), self.interpolant, self.edge_mode,
            self.constant))

    def __eq__(self, other):
        if self is other: return True
        if not (isinstance(other, LookupTable2D) and
                np.array_equal(self.x,other.x) and
                np.array_equal(self.y,other.y) and
                np.array_equal(self.f,other.f) and
                self.interpolant == other.interpolant and
                self.edge_mode == other.edge_mode and
                self.constant == other.constant):
            return False
        else:
            if self.interpolant == 'spline':
                return (np.array_equal(self.dfdx, other.dfdx) and
                        np.array_equal(self.dfdy, other.dfdy) and
                        np.array_equal(self.d2fdxdy, other.d2fdxdy))
            return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.LookupTable2D", tuple(self.x.ravel()), tuple(self.y.ravel()),
                               tuple(self.f.ravel()), self.interpolant, self.edge_mode,
                               self.constant,
                               tuple(self.dfdx.ravel()) if self.dfdx is not None else None,
                               tuple(self.dfdy.ravel()) if self.dfdy is not None else None,
                               tuple(self.d2fdxdy.ravel()) if self.d2fdxdy is not None else None))
        return self._hash

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_tab',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d


def _LookupTable2D(x, y, f, interpolant, edge_mode, constant,
                   dfdx=None, dfdy=None, d2fdxdy=None,
                   x0=None, y0=None, xperiod=None, yperiod=None):
    """Make a `LookupTable2D` but without using any of the sanity checks or array manipulation used
    in the normal initializer.
    """
    ret = LookupTable2D.__new__(LookupTable2D)
    ret.x = x
    ret.y = y
    ret.f = f
    ret.interpolant = interpolant
    ret.edge_mode = edge_mode
    ret.constant = constant
    ret.dfdx = dfdx
    ret.dfdy = dfdy
    ret.d2fdxdy = d2fdxdy
    ret.x0 = x0
    ret.y0 = y0
    ret.xperiod = xperiod
    ret.yperiod = yperiod
    if interpolant in ('nearest', 'linear', 'ceil', 'floor', 'spline'):
        ret._interp2d = None
    else:
        ret._interp2d = convert_interpolant(interpolant)
    return ret
