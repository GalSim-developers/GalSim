# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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

Also, a simple 2D table for uniformly gridded input data: LookupTable2D.
"""
from . import _galsim
import galsim


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
    @param interpolant   The interpolant to use, with the options being 'floor', 'ceil',
                         'linear' and 'spline'. [default: 'spline']
    @param x_log         Set to True if you wish to interpolate using log(x) rather than x.  Note
                         that all inputs / outputs will still be x, it's just a question of how the
                         interpolation is done. [default: False]
    @param f_log         Set to True if you wish to interpolate using log(f) rather than f.  Note
                         that all inputs / outputs will still be f, it's just a question of how the
                         interpolation is done. [default: False]
    """
    def __init__(self, x=None, f=None, file=None, interpolant=None, x_log=False, f_log=False):
        import numpy as np
        self.x_log = x_log
        self.f_log = f_log
        self.file = file

        # read in from file if a filename was specified
        if file:
            if x is not None or f is not None:
                raise ValueError("Cannot provide both file _and_ x,f for LookupTable")
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
        x = np.asarray(x).astype(float)
        f = np.asarray(f).astype(float)
        self.x = x
        self.f = f

        # check for proper interpolant
        if interpolant is None:
            interpolant = 'spline'
        else:
            if interpolant not in ['spline', 'linear', 'ceil', 'floor']:
                raise ValueError("Unknown interpolant: %s" % interpolant)
        self.interpolant = interpolant

        # make and store table
        if x_log:
            if np.any(np.array(x) <= 0.):
                raise ValueError("Cannot interpolate in log(x) when table contains x<=0!")
            x = np.log(x)
        if f_log:
            if np.any(np.array(f) <= 0.):
                raise ValueError("Cannot interpolate in log(f) when table contains f<=0!")
            f = np.log(f)

        # table is the thing the does the actual work.  It is a C++ Table object, wrapped
        # as _LookupTable.
        self.table = _galsim._LookupTable(x, f, interpolant)

    @property
    def x_min(self): return min(self.x)
    @property
    def x_max(self): return max(self.x)
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
        import numpy as np
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
                f = np.zeros_like(x)
                for i in xrange(x.shape[0]):
                    f[i,:] = np.fromiter((self.table(float(q)) for q in x[i,:]), dtype='float')
            else:
                f = np.fromiter((self.table(float(q)) for q in x), dtype='float')
        # option 2: a tuple
        elif isinstance(x, tuple):
            f = [ self.table(q) for q in x ]
            f = tuple(f)
        # option 3: a list
        elif isinstance(x, list):
            f = [ self.table(q) for q in x ]
        # option 4: a single value
        else:
            f = self.table(x)

        if self.f_log:
            f = np.exp(f)
        return f

    def getArgs(self):
        args = self.table.getArgs()
        if self.x_log:
            import numpy as np
            return np.exp(args)
        else:
            return args

    def getVals(self):
        vals = self.table.getVals()
        if self.f_log:
            import numpy as np
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
        import numpy as np
        return (isinstance(other, LookupTable) and
                np.array_equal(self.x,other.x) and
                np.array_equal(self.f,other.f) and
                self.x_log == other.x_log and
                self.f_log == other.f_log and
                self.interpolant == other.interpolant)
    def __ne__(self, other): return not self.__eq__(other)

    def __repr__(self):
        return 'galsim.LookupTable(x=array(%r), f=array(%r), x_log=%r, f_log=%r, interpolant=%r)'%(
            self.x.tolist(), self.f.tolist(), self.x_log, self.f_log, self.interpolant)

    def __str__(self):
        if self.file is not None:
            return 'galsim.LookupTable(file=%r, interpolant=%r)'%(
                self.file, self.interpolant)
        else:
            return 'galsim.LookupTable(x=[%s,..,%s], f=[%s,...,%s], interpolant=%r)'%(
                self.x[0], self.x[-1], self.f[0], self.f[-1], self.interpolant)

    def __hash__(self): return hash(repr(self))


# A function to enable pickling of tables
_galsim._LookupTable.__getinitargs__ = lambda self: \
        (self.getArgs(), self.getVals(), self.getInterp())
_galsim._LookupTable.__repr__ = lambda self: \
        'galsim._galsim._LookupTable(array(%r), array(%r), %r)'%(
            self.getArgs(), self.getVals(), self.getInterp())
_galsim._LookupTable.__eq__ = lambda self, other: repr(self) == repr(other)
_galsim._LookupTable.__ne__ = lambda self, other: not self.__eq__(other)
_galsim._LookupTable.__hash__ = lambda self: hash(repr(self))


class LookupTable2D(object):
    """
    LookupTable2D represents a 2-dimensional lookup table to store function values that may be slow
    to calculate, for which interpolating from a lookup table is sufficiently accurate.  A
    LookupTable2D is also useful for evaluating periodic 2-d functions given samples from a single
    period.

    A LookupTable2D representing the function f(x, y) may be constructed from initial offsets in
    both dimensions `x0` and `y0`, step sizes in both dimensions `dx` and `dy`, and an array of
    function values `f` (the max values of `x` and `y` are determined automatically from the shape
    of the array `f`).

    The default interpolation method is a cubic spline.  Other choices for the interpolant are the
    same as for `InterpolatedImage`:
      - 'nearest'
      - 'linear'
      - 'quintic'
      - 'sinc', this one may not work well with edge wrapping since it has a large kernel footprint.
      - 'lanczosN', where N is the order of the Lanczos interpolant

    The `edge_mode` keyword describes how to handle extrapolation beyond the initial input range.
    Possibilities include:
      - 'none': do nothing, silently allow extrapolation, which will return all zeros at positions
                beyond the combined extent of the initial range and interpolant kernel footprint.
      - 'warn': allow extrapolation, but issue a warning whenever a value beyond the initial range
                is requested.
      - 'wrap': infinitely wrap the initial range in both directions.

    Three methods are available to evaluate new function values with the lookup table:

      - the () operator, i.e.:

        > tab2d = LookupTable2D(...)
        > val = tab2d(x, y)

        The () operator has flexible input; the following are equivalent to the above:

        > val = tab2d(galsim.PositionD(x, y))
        > val = tab2d(y=y, x=x)

      - the at() method, which is similar to the () operator, but slightly faster since the input
        argument types do not need to be dynamically inferred:

        > val = tab2d.at(x, y)

      - the eval_grid() method, which is optimized for evaluating the lookup table on a grid.

        > vals = tab2d.eval_grid(xmin, xmax, nx, ymin, ymax, ny)

    @param x0             The minimum `x` position of the table inputs
    @param y0             The minimum `y` position of the table inputs
    @param dx             The `x` spacing of initial table inputs
    @param dy             The `y` spacing of initial table inputs
    @param f              The input array of function values
    @param interpolant    Interpolant to use.  [Default: 'cubic']
    @param edge_mode      Keyword controlling how extrapolation beyond the input range is handled.
                          See above for details.  [Default: 'warn']
    """
    def __init__(self, x0=0.0, y0=0.0, dx=1.0, dy=1.0, f=None, interpolant=None,
                 edge_mode=None):
        import numpy as np
        if interpolant is None:
            interpolant = 'cubic'
        self.interpolant = interpolant
        if edge_mode is None:
            edge_mode = 'warn'
        if edge_mode not in {'warn', 'wrap', 'none'}:
            raise ValueError("Unknown edge_mode")
        self.edge_mode = edge_mode

        self.f = np.array(f)
        ny, nx = self.f.shape
        self.xmin = x0
        self.xmax = x0 + (nx-1)*dx
        self.ymin = y0
        self.ymax = y0 + (ny-1)*dy

        xorigin = x0
        yorigin = y0

        if self.edge_mode == 'wrap':
            # Need to extend the input grid by a few columns/rows here to make the interpolation
            # work near the edges.  We wrap 3 rows/columns since the quintic interpolant footprint
            # is 5x5.  Handle extrapolations outside of the initial input footprint using modular
            # arithmetic inside the __call__(), at(), and eval_grid() methods.

            # wrap 3 rows on top/bottom edges
            self.f = np.vstack([self.f[-3:, :], self.f, self.f[:3, :]])
            # wrap 3 columns (including new rows) on left/right edges
            self.f = np.hstack([self.f[:, -3:], self.f, self.f[:, :3]])
            # Grid repeats over pre-extended array size.
            # Note that xrepeat != (xmax - xmin)  !!!  They're different by amount dx.
            self.xrepeat = nx * dx
            self.yrepeat = ny * dy
            # adjust origin for new array size and range
            xorigin -= 3*dx
            yorigin -= 3*dy
            nx += 6
            ny += 6

        # JM - In principle, we can integrate the offset into the wcs too with an
        # AffineTransform object.  I haven't figured out how to actually take advantage of that
        # when using drawImage, though, so for now I'm handling the origin offset manually, and the
        # "local" wcs (the pixel shape part) through the wcs framework.
        wcs = galsim.wcs.JacobianWCS(dx, 0.0, 0.0, dy)

        img = galsim.ImageD(self.f, wcs=wcs)
        self._ii = galsim.InterpolatedImage(
            img, x_interpolant=self.interpolant, normalization='sb', calculate_stepk=False,
            calculate_maxk=False, pad_factor=1,
        ).shift(xorigin+0.5*(nx-1)*dx, yorigin+0.5*(ny-1)*dy)

    def _wrap_pos(self, pos):
        x = (pos.x-self.xmin) % self.xrepeat + self.xmin
        y = (pos.y-self.ymin) % self.yrepeat + self.ymin
        return galsim.PositionD(x, y)

    def __call__(self, *args, **kwargs):
        """Interpolate/extrapolate the LookupTable2D to get `f(x, y)` at some `(x, y)` position.

        Multiple options are supported for the input position.  For example:

        > tab2d = LookupTable2D(...)
        > val = tab2d(x, y)
        > val = tab2d(galsim.PositionD(x, y))
        > val = tab2d(y=y, x=x)

        @returns the interpolated `f(x, y)` value.
        """
        pos = galsim.utilities.parse_pos_args(args, kwargs, 'x', 'y')
        if self.edge_mode == 'warn':
            import warnings
            if pos.x < self.xmin or pos.x > self.xmax or pos.y < self.ymin or pos.y > self.ymax:
                warnings.warn("Extrapolating beyond input range.")
        elif self.edge_mode == 'wrap':
            pos = self._wrap_pos(pos)
        return self._ii.xValue(pos)

    def at(self, x, y):
        """Interpolate/extrapolate the LookupTable2D to get `f(x, y)` at some `(x, y)` position.

        This method is slightly faster than the () operator since the input is more constrained.

        @param x   The `x` value for which `f(x, y)` should be evaluated
        @param y   The `y` value for which `f(x, y)` should be evaluated

        @returns   the interpolated `f(x, y)` value.
        """
        pos = galsim.PositionD(x, y)
        if self.edge_mode == 'warn':
            import warnings
            if pos.x < self.xmin or pos.x > self.xmax or pos.y < self.ymin or pos.y > self.ymax:
                warnings.warn("Extrapolating beyond input range.")
        elif self.edge_mode == 'wrap':
            pos = self._wrap_pos(pos)
        return self._ii.xValue(pos)

    def eval_grid(self, xmin, xmax, nx, ymin, ymax, ny):
        """Evaluate the LookupTable2D on a regularly spaced grid.  This method is significantly
        faster for grid evaluations than repeated calling the .at() method or the () operator.

        @param xmin  Minimum value of `x` for which to obtain `f(x, y)`
        @param xmax  Maximum value of `x` for which to obtain `f(x, y)`
        @param nx    Number of grid points in the `x` direction for which to obtain `f(x, y)`
        @param ymin  Minimum value of `y` for which to obtain `f(x, y)`
        @param ymax  Maximum value of `y` for which to obtain `f(x, y)`
        @param ny    Number of grid points in the `y` direction for which to obtain `f(x, y)`
        @returns     Array of `f(x, y)` values.
        """
        if self.edge_mode == 'wrap':
            return self._eval_grid_wrap(xmin, xmax, nx, ymin, ymax, ny)
        elif self.edge_mode == 'warn':
            import warnings
            if xmin < self.xmin or xmax > self.xmax or ymin < self.ymin or ymax > self.ymax:
                warnings.warn("Extrapolating beyond input range.")
        return self._eval_grid(xmin, xmax, nx, ymin, ymax, ny)

    def _eval_grid(self, xmin, xmax, nx, ymin, ymax, ny):
        # Assumes no extrapolation.  I.e., that xmin > self.xmin, xmax < self.xmax, etc.
        dx = (xmax-xmin)/(nx-1.0) if nx != 1 else 1e-12
        dy = (ymax-ymin)/(ny-1.0) if ny != 1 else 1e-12
        xmean = 0.5*(xmin+xmax)
        ymean = 0.5*(ymin+ymax)
        wcs = galsim.wcs.JacobianWCS(dx, 0.0, 0.0, dy)
        offset = (-xmean, -ymean)
        return self._ii.shift(offset).drawImage(nx=nx, ny=ny, method='sb', wcs=wcs).array

    def _eval_grid_wrap(self, xmin, xmax, nx, ymin, ymax, ny):
        # implement edge wrapping by repeatedly identifying grid cells that map back onto
        # the "unwrapped" input coordinates.
        import numpy as np
        out = np.empty((ny, nx), dtype=float)
        # Output grid spacing
        dx = (xmax-xmin)/(nx-1.0)
        dy = (ymax-ymin)/(ny-1.0)

        # find wrap # that extends just below xmin
        i = (xmin - self.xmin) // self.xrepeat
        xlo, xhi = _lohi(i, self.xmin, self.xrepeat)  # current cell range
        ix = 0  # lower index for current cell in output array
        while xlo < xmax:
            # find output x range within current wrap #
            xmaxtmp = min([((xhi - xmin) // dx) * dx + xmin, xmax])
            xmintmp = max([((xlo - xmin) // dx + 1) * dx + xmin, xmin])
            nxtmp = int(round((xmaxtmp - xmintmp) / dx)) + 1

            # find wrap # that extends just below ymin
            j = (ymin - self.ymin) // self.yrepeat
            ylo, yhi = _lohi(j, self.ymin, self.yrepeat)
            iy = 0
            while ylo < ymax:
                # find output y range within current wrap #
                ymaxtmp = min([((yhi - ymin) // dy) * dy + ymin, ymax])
                ymintmp = max([((ylo - ymin) // dy + 1) * dy + ymin, ymin])
                nytmp = int(round((ymaxtmp - ymintmp) / dy)) + 1

                # _eval_grid with appropriately unwrapped coordinates
                out[iy:iy+nytmp, ix:ix+nxtmp] = self._eval_grid(
                    xmintmp - i * self.xrepeat, xmaxtmp - i * self.xrepeat, nxtmp,
                    ymintmp - j * self.yrepeat, ymaxtmp - j * self.yrepeat, nytmp)

                # prepare for next wrap #
                j += 1
                ylo, yhi = _lohi(j, self.ymin, self.yrepeat)
                iy += nytmp
            i += 1
            xlo, xhi = _lohi(i, self.xmin, self.xrepeat)
            ix += nxtmp
        return out


def _lohi(i, x0, dx):
    return x0 + i * dx, x0 + (i+1) * dx
