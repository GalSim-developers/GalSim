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
""""@file table.py
A few adjustments to galsim.LookupTable at the Python layer, including the
addition of the docstring and few extra features.

Also, a simple 2D table for gridded input data: LookupTable2D.
"""
import numpy as np

from . import _galsim


class LookupTable(object):
    """
    LookupTable represents a lookup table to store function values that may be slow to calculate,
    for which interpolating from a lookup table is sufficiently accurate.

    A LookupTable may be constructed from two arrays (lists, tuples, or NumPy arrays of
    floats/doubles).

        >>> args = [...]
        >>> vals = []
        >>> for arg in args:
        ...     val = calculateVal(arg)
        ...     vals.append(val)
        >>> table = galsim.LookupTable(x=args,f=vals)

    Then you can use this table as a replacement for the slow calculation:

        >>> other_args = [...]
        >>> for arg in other_args:
        ...     val = table(arg)
        ...     [... use val ...]


    The default interpolation method is cubic spline interpolation.  This is usually the
    best choice, but we also provide three other options, which can be specified by
    the `interpolant` kwarg.  The choices are 'floor', 'ceil', 'linear' and 'spline':

    - 'floor' takes the value from the previous argument in the table.
    - 'ceil' takes the value from the next argument in the table.
    - 'nearest' takes the value from the nearest argument in the table.
    - 'linear' does linear interpolation between these two values.
    - 'spline' uses a cubic spline interpolation, so the interpolated values are smooth at
      each argument in the table.

    Another option is to read in the values from an ascii file.  The file should have two
    columns of numbers, which are taken to be the `x` and `f` values.

    The user can also opt to interpolate in log(x) and/or log(f), though this is not the default.
    It may be a wise choice depending on the particular function, e.g., for a nearly power-law
    f(x) (or at least one that is locally power-law-ish for much of the x range) then it might
    be a good idea to interpolate in log(x) and log(f) rather than x and f.

    @param x             The list, tuple, or NumPy array of `x` values (floats, doubles, or ints,
                         which get silently converted to floats for the purpose of interpolation).
                         [Either `x` and `f` or `file` is required.]
    @param f             The list, tuple, or NumPy array of `f(x)` values (floats, doubles, or ints,
                         which get silently converted to floats for the purpose of interpolation).
                         [Either `x` and `f` or `file` is required.]
    @param file          A file from which to read the `(x,f)` pairs. [Either `x` and `f`, or `file`
                         is required]
    @param interpolant   The interpolant to use, with the options being 'floor', 'ceil', 'nearest',
                         'linear' and 'spline'. [default: 'spline']
    @param x_log         Set to True if you wish to interpolate using log(x) rather than x.  Note
                         that all inputs / outputs will still be x, it's just a question of how the
                         interpolation is done. [default: False]
    @param f_log         Set to True if you wish to interpolate using log(f) rather than f.  Note
                         that all inputs / outputs will still be f, it's just a question of how the
                         interpolation is done. [default: False]
    """
    def __init__(self, x=None, f=None, file=None, interpolant=None, x_log=False, f_log=False):
        self.x_log = x_log
        self.f_log = f_log
        self.file = file

        # read in from file if a filename was specified
        if file:
            if x is not None or f is not None:
                raise ValueError("Cannot provide both file _and_ x,f for LookupTable")
            # We don't require pandas as a dependency, but if it's available, this is much faster.
            # cf. http://stackoverflow.com/questions/15096269/the-fastest-way-to-read-input-in-python
            CParserError = AttributeError # In case we don't get to the line below where we import
                                          # it from pandas.parser
            try:
                import pandas
                try:
                    # version >= 0.20
                    from pandas.io.common import CParserError
                except ImportError:
                    # version < 0.20
                    from pandas.parser import CParserError
                data = pandas.read_csv(file, comment='#', delim_whitespace=True, header=None)
                data = data.values.transpose()
            except (ImportError, AttributeError, CParserError):
                data = np.loadtxt(file).transpose()
            if data.shape[0] != 2:
                raise ValueError("File %s provided for LookupTable does not have 2 columns"%file)
            x=data[0]
            f=data[1]
        else:
            if x is None or f is None:
                raise ValueError("Must specify either file or x,f for LookupTable")

        # turn x and f into numpy arrays so that all subsequent math is possible (unlike for
        # lists, tuples).  Also make sure the dtype is float
        x = np.array(x, dtype=float)
        f = np.array(f, dtype=float)
        self.x = x
        self.f = f

        # check for proper interpolant
        if interpolant is None:
            interpolant = 'spline'
        else:
            if interpolant not in ['spline', 'linear', 'ceil', 'floor', 'nearest']:
                raise ValueError("Unknown interpolant: %s" % interpolant)
        self.interpolant = interpolant

        # make and store table
        if x_log:
            if np.any(x <= 0.):
                raise ValueError("Cannot interpolate in log(x) when table contains x<=0!")
            x = np.log(x)
        if f_log:
            if np.any(f <= 0.):
                raise ValueError("Cannot interpolate in log(f) when table contains f<=0!")
            f = np.log(f)

        # Sanity checks
        if len(x) != len(f):
            raise ValueError("Input array lengths don't match")
        if interpolant == 'spline' and len(x) < 3:
            raise ValueError("Input arrays too small to spline interpolate")
        if interpolant in ['linear', 'ceil', 'floor', 'nearest'] and len(x) < 2:
            raise ValueError("Input arrays too small to interpolate")

        # table is the thing the does the actual work.  It is a C++ Table object, wrapped
        # as _LookupTable.  Note x must be sorted.
        s = np.argsort(x)
        self.table = _galsim._LookupTable(x[s], f[s], interpolant)

        # Get the min/max x values, making sure to account properly for x_log.
        self._x_min = self.table.argMin()
        self._x_max = self.table.argMax()
        if x_log:
            self._x_min = np.exp(self._x_min)
            self._x_max = np.exp(self._x_max)

    @property
    def x_min(self): return self._x_min
    @property
    def x_max(self): return self._x_max
    @property
    def n_x(self): return len(self.x)

    def __call__(self, x):
        """Interpolate the LookupTable to get `f(x)` at some `x` value(s).

        When the LookupTable object is called with a single argument, it returns the value at that
        argument.  An exception will be thrown automatically by the _LookupTable class if the `x`
        value is outside the range of the original tabulated values.  The value that is returned is
        the same type as that provided as an argument, e.g., if a single value `x` is provided then
        a single value of `f` is returned; if a tuple of `x` values is provided then a tuple of `f`
        values is returned; and so on.  Even if interpolation was done using the `x_log` option,
        the user should still provide `x` rather than `log(x)`.

        @param x        The `x` value(s) for which `f(x)` should be calculated via interpolation on
                        the original `(x,f)` lookup table.  `x` can be a single float/double, or a
                        tuple, list, or arbitrarily shaped 1- or 2-dimensional NumPy array.

        @returns the interpolated `f(x)` value(s).
        """
        # first, keep track of whether interpolation was done in x or log(x)
        if self.x_log:
            if np.any(np.array(x) <= 0.):
                raise ValueError("Cannot interpolate x<=0 when using log(x) interpolation.")
            x = np.log(x)

        # figure out what we received, and return the same thing
        # option 1: a NumPy array
        if isinstance(x, np.ndarray):
            dimen = len(x.shape)
            if dimen > 2:
                raise ValueError("Arrays with dimension larger than 2 not allowed!")
            elif dimen == 2:
                f = np.empty_like(x.ravel(), dtype=float)
                self.table.interpMany(x.astype(float,copy=False).ravel(),f)
                f = f.reshape(x.shape)
            else:
                f = np.empty_like(x, dtype=float)
                self.table.interpMany(x.astype(float,copy=False),f)
        # option 2: a tuple
        elif isinstance(x, tuple):
            f = np.empty_like(x, dtype=float)
            self.table.interpMany(np.array(x, dtype=float),f)
            f = tuple(f)
        # option 3: a list
        elif isinstance(x, list):
            f = np.empty_like(x, dtype=float)
            self.table.interpMany(np.array(x, dtype=float),f)
            f = list(f)
        # option 4: a single value
        else:
            f = self.table(x)

        if self.f_log:
            f = np.exp(f)
        return f

    def getArgs(self):
        args = self.table.getArgs()
        if self.x_log:
            return np.exp(args)
        else:
            return args

    def getVals(self):
        vals = self.table.getVals()
        if self.f_log:
            return np.exp(vals)
        else:
            return vals

    def getInterp(self):
        return self.table.getInterp()

    def isLogX(self):
        return self.x_log

    def isLogF(self):
        return self.f_log

    def __eq__(self, other):
        return (isinstance(other, LookupTable) and
                np.array_equal(self.x,other.x) and
                np.array_equal(self.f,other.f) and
                self.x_log == other.x_log and
                self.f_log == other.f_log and
                self.interpolant == other.interpolant)
    def __ne__(self, other): return not self.__eq__(other)

    def __hash__(self):
        # Cache this in case self.x, self.f are long.
        if not hasattr(self, '_hash'):
            self._hash = hash(("galsim.LookupTable", tuple(self.x), tuple(self.f), self.x_log,
                               self.f_log, self.interpolant))
        return self._hash


    def __repr__(self):
        return 'galsim.LookupTable(x=array(%r), f=array(%r), x_log=%r, f_log=%r, interpolant=%r)'%(
            self.x.tolist(), self.f.tolist(), self.x_log, self.f_log, self.interpolant)

    def __str__(self):
        if self.file is not None:
            return 'galsim.LookupTable(file=%r, interpolant=%r)'%(
                self.file, self.interpolant)
        else:
            return 'galsim.LookupTable(x=[%s,...,%s], f=[%s,...,%s], interpolant=%r)'%(
                self.x[0], self.x[-1], self.f[0], self.f[-1], self.interpolant)

# A function to enable pickling of tables
_galsim._LookupTable.__getinitargs__ = lambda self: \
        (self.getArgs(), self.getVals(), self.getInterp())
_galsim._LookupTable.__repr__ = lambda self: \
        'galsim._galsim._LookupTable(array(%r), array(%r), %r)'%(
            self.getArgs(), self.getVals(), self.getInterp())

def _LookupTable_eq(self, other):
    return (isinstance(other, _galsim._LookupTable) and
            self.getArgs() == other.getArgs() and
            self.getVals() == other.getVals() and
            self.getInterp() == other.getInterp())

def _LookupTable_hash(self):
    return hash(("_galsim._LookupTable", tuple(self.getArgs()), tuple(self.getVals()),
                 self.getInterp()))

_galsim._LookupTable.__eq__ = _LookupTable_eq
_galsim._LookupTable.__ne__ = lambda self, other: not self.__eq__(other)
_galsim._LookupTable.__hash__ = _LookupTable_hash


class LookupTable2D(object):
    """
    LookupTable2D represents a 2-dimensional lookup table to store function values that may be slow
    to calculate, for which interpolating from a lookup table is sufficiently accurate.  A
    LookupTable2D is also useful for evaluating periodic 2-d functions given samples from a single
    period.

    A LookupTable2D representing the function f(x, y) may be constructed from a list or array of `x`
    values, a list or array of `y` values, and a 2D array of function evaluations at all
    combinations of x and y values.  For instance:

        >>> x = np.arange(5)
        >>> y = np.arange(8)
        >>> z = x[:, np.newaxis] + y  # function is x + y, dimensions of z are (5, 8)
        >>> tab2d = galsim.LookupTable2D(x, y, z)

    To evaluate new function values with the lookup table, use the () operator:

        >>> print tab2d(2.2, 3.3)
        5.5

    The () operator can also accept sequences (lists, tuples, numpy arrays, ...) for the x and y
    arguments at which to evaluate the LookupTable2D.  The x and y sequences should have the same
    length in this case, which will also be the length of the output sequence.

        >>> print tab2d([1, 2], [3, 4])
        [ 4.  6.]

    The default interpolation method is linear.  Other choices for the interpolant are:
      - 'floor'
      - 'ceil'
      - 'nearest'

        >>> tab2d = galsim.LookupTable2D(x, y, z, interpolant='floor')
        >>> tab2d(2.2, 3.7)
        5.0
        >>> tab2d = galsim.LookupTable2D(x, y, z, interpolant='ceil')
        >>> tab2d(2.2, 3.7)
        7.0
        >>> tab2d = galsim.LookupTable2D(x, y, z, interpolant='nearest')
        >>> tab2d(2.2, 3.7)
        6.0

    The `edge_mode` keyword describes how to handle extrapolation beyond the initial input range.
    Possibilities include:
      - 'raise': raise an exception.  (This is the default.)
      - 'constant': Return a constant specified by the `constant` keyword.
      - 'wrap': infinitely wrap the initial range in both directions.
    In order for LookupTable2D to determine the wrapping period when edge_mode='wrap', either the
    x and y grid points need to be equally spaced (in which case the x-period is inferred as
    len(x)*(x[1]-x[0]) and similarly for y), or the first/last row/column of f must be identical,
    in which case the x-period is inferred as x[-1] - x[0].  (If both conditions are satisfied
    (equally-spaced x and y and identical first/last row/column of f, then the x-period is inferred
    as len(x)*(x[1]-x[0])).

        >>> x = np.arange(5)
        >>> y = np.arange(8)
        >>> z = x[:, np.newaxis] + y  # function is x + y, dimensions of z is (5, 8)
        >>> tab2d = galsim.LookupTable2D(x, y, z, edge_mode='raise')
        >>> tab2d(7, 7)
        ValueError: Extrapolating beyond input range.

        >>> tab2d = galsim.LookupTable2D(x, y, z, edge_mode='constant', constant=1.0)
        1.0

        >>> tab2d = galsim.LookupTable2D(x, y, z, edge_mode='wrap')
        ValueError: Cannot wrap `f` array with unequal first/last column/row.

    We extend the x and y arrays with a uniform spacing, though any monotonic spacing would work.
    Note that the [(0,1), (0,1)] argument in np.pad below extends the z array by 0 rows/columns in
    the leading direction, and 1 row/column in the trailing direction.

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

    @param x              Strictly increasing array of `x` positions at which to create table.
    @param y              Strictly increasing array of `y` positions at which to create table.
    @param f              Nx by Ny input array of function values.
    @param interpolant    Interpolant to use.  One of 'floor', 'ceil', 'nearest', or 'linear'.
                          [Default: 'linear']
    @param edge_mode      Keyword controlling how extrapolation beyond the input range is handled.
                          See above for details.  [Default: 'raise']
    @param constant       A constant to return when extrapolating beyond the input range and
                          `edge_mode='constant'`.  [Default: 0]
    """
    def __init__(self, x, y, f, interpolant='linear', edge_mode='raise', constant=0):
        if edge_mode not in ['raise', 'wrap', 'constant']:
            raise ValueError("Unknown edge_mode: {:0}".format(edge_mode))
        self.edge_mode = edge_mode

        self.x = np.ascontiguousarray(x, dtype=float)
        self.y = np.ascontiguousarray(y, dtype=float)
        self.f = np.ascontiguousarray(f, dtype=float)

        dx = np.diff(self.x)
        dy = np.diff(self.y)

        if not (all(dx > 0) and all(dy > 0)):
            raise ValueError("x and y input grids are not strictly increasing.")

        fshape = self.f.shape
        if fshape != (len(x), len(y)):
            raise ValueError("Shape of `f` must be (len(`x`), len(`y`)).")

        self.interpolant = interpolant
        self.edge_mode = edge_mode
        self.constant = float(constant)

        if self.edge_mode == 'wrap':
            # Can wrap if x and y arrays are equally spaced ...
            if np.allclose(dx, dx[0]) and np.allclose(dy, dy[0]):
                # Underlying Table2D requires us to extend x, y, and f.
                self.x = np.append(self.x, self.x[-1]+dx[0])
                self.y = np.append(self.y, self.y[-1]+dy[0])
                self.f = np.pad(self.f, [(0,1), (0,1)], mode='wrap')
            if (all(self.f[0] == self.f[-1]) and all(self.f[:,0] == self.f[:,-1])):
                self.xperiod = self.x[-1] - self.x[0]
                self.yperiod = self.y[-1] - self.y[0]
            else:
                raise ValueError("Cannot use edge_mode='wrap' unless either x and y are equally "
                                 "spaced or first/last row/column of f are identical.")

        self.table = _galsim._LookupTable2D(self.x, self.y, self.f, self.interpolant)

    def _inbounds(self, x, y):
        """Return whether or not *all* coords specified by x and y are in bounds of the original
        interpolated array."""
        return (np.min(x) >= self.x[0] and np.max(x) <= self.x[-1] and
                np.min(y) >= self.y[0] and np.max(y) <= self.y[-1])

    def _wrap_args(self, x, y):
        """Wrap points back into the fundamental period."""
        return ((x-self.x[0]) % self.xperiod + self.x[0],
                (y-self.y[0]) % self.yperiod + self.y[0])

    def _call_raise(self, x, y):
        if not self._inbounds(x, y):
            raise ValueError("Extrapolating beyond input range.")

        from numbers import Real
        if isinstance(x, Real):
            return self.table(x, y)
        else:
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
            shape = x.shape
            x = x.ravel()
            y = y.ravel()
            f = np.empty_like(x, dtype=float)
            self.table.interpMany(x, y, f)
            f = f.reshape(shape)
            return f

    def _call_wrap(self, x, y):
        x, y = self._wrap_args(x, y)
        return self._call_raise(x, y)

    def _call_constant(self, x, y):
        from numbers import Real
        if isinstance(x, Real):
            if self._inbounds(x, y):
                return self.table(x, y)
            else:
                return self.constant
        else:
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
            shape = x.shape
            x = x.ravel()
            y = y.ravel()
            f = np.empty_like(x)
            f.fill(self.constant)
            good = ((x >= self.x[0]) & (x <= self.x[-1]) &
                    (y >= self.y[0]) & (y <= self.y[-1]))
            tmp = np.empty((sum(good),), dtype=float)
            self.table.interpMany(x[good], y[good], tmp)
            f[good] = tmp
            f = f.reshape(shape)
            return f

    def __call__(self, x, y):
        if self.edge_mode == 'raise':
            return self._call_raise(x, y)
        elif self.edge_mode == 'wrap':
            return self._call_wrap(x, y)
        else: # constant
            return self._call_constant(x, y)

    def _gradient_raise(self, x, y):
        if not self._inbounds(x, y):
            raise ValueError("Extrapolating beyond input range.")

        try:
            xx = float(x)
            yy = float(y)
            return self.table.gradient(xx, yy)
        except TypeError:
            dfdx = np.empty_like(x)
            dfdy = np.empty_like(x)
            self.table.gradientMany(x.ravel(), y.ravel(), dfdx.ravel(), dfdy.ravel())
            return dfdx, dfdy

    def _gradient_wrap(self, x, y):
        x, y = self._wrap_args(x, y)
        return self._gradient_raise(x, y)

    def _gradient_constant(self, x, y):
        from numbers import Real
        if isinstance(x, Real):
            if self._inbounds(x, y):
                return self.table.gradient(x, y)
            else:
                return 0.0, 0.0
        else:
            dfdx = np.empty_like(x)
            dfdy = np.empty_like(y)
            dfdx.fill(0.0)
            dfdy.fill(0.0)
            good = ((x >= self.x[0]) & (x <= self.x[-1]) &
                    (y >= self.y[0]) & (y <= self.y[-1]))
            tmp1 = np.empty((np.sum(good),), dtype=x.dtype)
            tmp2 = np.empty((np.sum(good),), dtype=x.dtype)
            self.table.gradientMany(x[good], y[good], tmp1, tmp2)
            dfdx[good] = tmp1
            dfdy[good] = tmp2
            return dfdx, dfdy

    def gradient(self, x, y):
        """Calculate the gradient of the function at an arbitrary point or points.

        @param x        Either a single x value or an array of x values at which to compute
                        the gradient.
        @param y        Either a single y value or an array of y values at which to compute
                        the gradient.

        @returns A tuple of (dfdx, dfdy) where dfdx, dfdy are single values (if x,y were single
        values) or numpy arrays.
        """
        if self.edge_mode == 'raise':
            return self._gradient_raise(x, y)
        elif self.edge_mode == 'wrap':
            return self._gradient_wrap(x, y)
        else: # constant
            return self._gradient_constant(x, y)

    def __str__(self):
        return ("galsim.LookupTable2D(x=[%s,...,%s], y=[%s,...,%s], "
                "f=[[%s,...,%s],...,[%s,...,%s]], interpolant=%r, edge_mode=%r)"%(
            self.x[0], self.x[-1], self.y[0], self.y[-1],
            self.f[0,0], self.f[0,-1], self.f[-1,0], self.f[-1,-1],
            self.interpolant, self.edge_mode))

    def __repr__(self):
        return ("galsim.LookupTable2D(x=array(%r), y=array(%r), "
                "f=array(%r), interpolant=%r, edge_mode=%r)"%(
            self.x.tolist(), self.y.tolist(), self.f.tolist(), self.interpolant, self.edge_mode))

    def __eq__(self, other):
        return (isinstance(other, LookupTable2D) and
                self.table == other.table and
                self.edge_mode == other.edge_mode)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("galsim._galsim._LookupTable2D", self.table, self.edge_mode))

def _LookupTable2D_eq(self, other):
    return (isinstance(other, _galsim._LookupTable2D)
            and np.array_equal(self.getXArgs(), other.getXArgs())
            and np.array_equal(self.getYArgs(), other.getYArgs())
            and np.array_equal(self.getVals(), other.getVals())
            and self.getInterp() == other.getInterp())

def _LookupTable2D_str(self):
    x = self.getXArgs()
    y = self.getYArgs()
    f = self.getVals()
    return ("galsim._galsim._LookupTable2D(x=[%s,...,%s], y=[%s,...,%s], "
            "f=[[%s,...,%s],...,[%s,...,%s]], interpolant=%r)"%(
            x[0], x[-1], y[0], y[-1], f[0,0], f[0,-1], f[-1,0], f[-1,-1], self.getInterp()))

_galsim._LookupTable2D.__getinitargs__ = lambda self: \
        (self.getXArgs(), self.getYArgs(), self.getVals(), self.getInterp())
_galsim._LookupTable2D.__eq__ = _LookupTable2D_eq
_galsim._LookupTable2D.__hash__ = lambda self: \
        hash(("_galsim._LookupTable2D", tuple(self.getXArgs()), tuple(self.getYArgs()),
              tuple(np.array(self.getVals()).ravel()), self.getInterp()))
_galsim._LookupTable2D.__str__ = _LookupTable2D_str
_galsim._LookupTable2D.__repr__ = lambda self: \
        'galsim._galsim._LookupTable2D(array(%r), array(%r), array(%r), %r)'%(
        self.getXArgs().tolist(), self.getYArgs().tolist(), self.getVals().tolist(),
        self.getInterp())
