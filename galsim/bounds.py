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

import math

from . import _galsim
from .position import Position, PositionI, PositionD, _PositionD, _PositionI
from .errors import GalSimUndefinedBoundsError

class Bounds(object):
    """A class for representing image bounds as 2D rectangles.

    Bounds is a base class for two slightly different kinds of bounds:
    `BoundsD` describes bounds with floating point values in x and y.
    `BoundsI` described bounds with integer values in x and y.

    The bounds are stored as four numbers in each instance, (xmin, xmax, ymin, ymax), with an
    additional boolean switch to say whether or not the Bounds rectangle has been defined.  The
    rectangle is undefined if the min value > the max value in either direction.

    *Initialization*:

    A `BoundsI` or `BoundsD` instance can be initialized in a variety of ways.  The most direct is
    via four scalars::

        >>> bounds = galsim.BoundsD(xmin, xmax, ymin, ymax)
        >>> bounds = galsim.BoundsI(imin, imax, jmin, jmax)

    In the `BoundsI` example above, ``imin``, ``imax``, ``jmin`` and ``jmax`` must all be integers
    to avoid a TypeError exception.

    Another way to initialize a `Bounds` instance is using two `Position` instances, the first
    for ``(xmin,ymin)`` and the second for ``(xmax,ymax)``::

        >>> bounds = galsim.BoundsD(galsim.PositionD(xmin, ymin), galsim.PositionD(xmax, ymax))
        >>> bounds = galsim.BoundsI(galsim.PositionI(imin, jmin), galsim.PositionI(imax, jmax))

    In both the examples above, the I/D type of `PositionI`/`PositionD` must match that of
    `BoundsI`/`BoundsD`.

    Finally, there are a two ways to lazily initialize a bounds instance with ``xmin = xmax``,
    ``ymin = ymax``, which will have an undefined rectangle and the instance method isDefined()
    will return False.  The first sets ``xmin = xmax = ymin = ymax = 0``::

        >>> bounds = galsim.BoundsD()
        >>> bounds = galsim.BoundsI()

    The second method sets both upper and lower rectangle bounds to be equal to some position::

        >>> bounds = galsim.BoundsD(galsim.PositionD(xmin, ymin))
        >>> bounds = galsim.BoundsI(galsim.PositionI(imin, jmin))

    Once again, the I/D type of `PositionI`/`PositionD` must match that of `BoundsI`/`BoundsD`.

    For the latter two initializations, you would typically then add to the bounds with::

        >>> bounds += pos1
        >>> bounds += pos2
        >>> [etc.]

    Then the bounds will end up as the bounding box of all the positions that were added to it.

    You can also find the intersection of two bounds with the & operator::

        >>> overlap = bounds1 & bounds2

    This is useful for adding one image to another when part of the first image might fall off
    the edge of the other image::

        >>> overlap = stamp.bounds & image.bounds
        >>> image[overlap] += stamp[overlap]

    """
    def __init__(self):
        raise NotImplementedError("Cannot instantiate the base class. "
                                  "Use either BoundsD or BoundsI.")

    def _parse_args(self, *args, **kwargs):
        if len(kwargs) == 0:
            if len(args) == 4:
                self._isdefined = True
                self.xmin, self.xmax, self.ymin, self.ymax = args
            elif len(args) == 0:
                self._isdefined = False
                self.xmin = self.xmax = self.ymin = self.ymax = 0
            elif len(args) == 1:
                if isinstance(args[0], (Bounds, _galsim.BoundsD, _galsim.BoundsI)):
                    self._isdefined = True
                    self.xmin = args[0].xmin
                    self.xmax = args[0].xmax
                    self.ymin = args[0].ymin
                    self.ymax = args[0].ymax
                elif isinstance(args[0], (Position, _galsim.PositionD, _galsim.PositionI)):
                    self._isdefined = True
                    self.xmin = self.xmax = args[0].x
                    self.ymin = self.ymax = args[0].y
                else:
                    raise TypeError("Single argument to %s must be either a Bounds or a Position"%(
                                    self.__class__.__name__))
                self._isdefined = True
            elif len(args) == 2:
                if (isinstance(args[0], (Position, _galsim.PositionD, _galsim.PositionI)) and
                    isinstance(args[1], (Position, _galsim.PositionD, _galsim.PositionI))):
                    self._isdefined = True
                    self.xmin = min(args[0].x, args[1].x)
                    self.xmax = max(args[0].x, args[1].x)
                    self.ymin = min(args[0].y, args[1].y)
                    self.ymax = max(args[0].y, args[1].y)
                else:
                    raise TypeError("Two arguments to %s must be Positions"%(
                                    self.__class__.__name__))
            else:
                raise TypeError("%s takes either 1, 2, or 4 arguments (%d given)"%(
                                self.__class__.__name__,len(args)))
        elif len(args) != 0:
            raise TypeError("Cannot provide both keyword and non-keyword arguments to %s"%(
                            self.__class__.__name__))
        else:
            try:
                self._isdefined = True
                self.xmin = kwargs.pop('xmin')
                self.xmax = kwargs.pop('xmax')
                self.ymin = kwargs.pop('ymin')
                self.ymax = kwargs.pop('ymax')
            except KeyError:
                raise TypeError("Keyword arguments, xmin, xmax, ymin, ymax are required for %s"%(
                                self.__class__.__name__))
            if kwargs:
                raise TypeError("Got unexpected keyword arguments %s"%kwargs.keys())

        if not (float(self.xmin) <= float(self.xmax) and float(self.ymin) <= float(self.ymax)):
            self._isdefined = False

    def area(self):
        """Return the area of the enclosed region.

        The area is a bit different for integer-type `BoundsI` and float-type `BoundsD` instances.
        For floating point types, it is simply ``(xmax-xmin)*(ymax-ymin)``.  However, for integer
        types, we add 1 to each size to correctly count the number of pixels being described by the
        bounding box.
        """
        return self._area()

    def withBorder(self, dx, dy=None):
        """Return a new `Bounds` object that expands the current bounds by the specified width.

        If two arguments are given, then these are separate dx and dy borders.
        """
        self._check_scalar(dx, "dx")
        if dy is None:
            dy = dx
        else:
            self._check_scalar(dy, "dy")
        return self.__class__(self.xmin-dx, self.xmax+dx, self.ymin-dy, self.ymax+dy)

    @property
    def origin(self):
        "The lower left position of the `Bounds`."
        return self._pos_class(self.xmin, self.ymin)

    @property
    def center(self):
        """The central position of the `Bounds`.

        For a `BoundsI`, this will return an integer `PositionI`, which will be above and/or to
        the right of the true center if the x or y ranges have an even number of pixels.

        For a `BoundsD`, this is equivalent to true_center.
        """
        if not self.isDefined():
            raise GalSimUndefinedBoundsError("center is invalid for an undefined Bounds")
        return self._center

    @property
    def true_center(self):
        """The central position of the `Bounds` as a `PositionD`.

        This is always (xmax + xmin)/2., (ymax + ymin)/2., even for integer `BoundsI`, where
        this may not necessarily be an integer `PositionI`.
        """
        if not self.isDefined():
            raise GalSimUndefinedBoundsError("true_center is invalid for an undefined Bounds")
        return _PositionD((self.xmax + self.xmin)/2., (self.ymax + self.ymin)/2.)

    def includes(self, *args):
        """Test whether a supplied ``(x,y)`` pair, `Position`, or `Bounds` lie within a defined
        `Bounds` rectangle of this instance.

        Examples::

            >>> bounds = galsim.BoundsD(0., 100., 0., 100.)
            >>> bounds.includes(50., 50.)
            True
            >>> bounds.includes(galsim.PositionD(50., 50.))
            True
            >>> bounds.includes(galsim.BoundsD(-50., -50., 150., 150.))
            False

        The type of the `PositionI`/`PositionD` and `BoundsI`/`BoundsD` instances (i.e. integer or
        float type) should match that of the bounds instance.
        """
        if len(args) == 1:
            if isinstance(args[0], Bounds):
                b = args[0]
                return (self.isDefined() and b.isDefined() and
                        self.xmin <= b.xmin and
                        self.xmax >= b.xmax and
                        self.ymin <= b.ymin and
                        self.ymax >= b.ymax)
            elif isinstance(args[0], Position):
                p = args[0]
                return (self.isDefined() and
                        self.xmin <= p.x <= self.xmax and
                        self.ymin <= p.y <= self.ymax)
            else:
                raise TypeError("Invalid argument %s"%args[0])
        elif len(args) == 2:
            x, y = args
            return (self.isDefined() and
                    self.xmin <= float(x) <= self.xmax and
                    self.ymin <= float(y) <= self.ymax)
        elif len(args) == 0:
            raise TypeError("include takes at least 1 argument (0 given)")
        else:
            raise TypeError("include takes at most 2 arguments (%d given)"%len(args))

    def expand(self, factor):
        "Grow the `Bounds` by the supplied factor about the center."
        dx = (self.xmax - self.xmin) * 0.5 * (factor-1.)
        dy = (self.ymax - self.ymin) * 0.5 * (factor-1.)
        if isinstance(self, BoundsI):
            dx = int(math.ceil(dx))
            dy = int(math.ceil(dy))
        return self.withBorder(dx,dy)

    def isDefined(self):
        "Test whether `Bounds` rectangle is defined."
        return self._isdefined

    def getXMin(self):
        "Get the value of xmin."
        return self.xmin

    def getXMax(self):
        "Get the value of xmax."
        return self.xmax

    def getYMin(self):
        "Get the value of ymin."
        return self.ymin

    def getYMax(self):
        "Get the value of ymax."
        return self.ymax

    def shift(self, delta):
        """Shift the `Bounds` instance by a supplied `Position`.

        Examples:

        The shift method takes either a `PositionI` or `PositionD` instance, which must match
        the type of the `Bounds` instance::

            >>> bounds = BoundsI(1,32,1,32)
            >>> bounds = bounds.shift(galsim.PositionI(3, 2))
            >>> bounds = BoundsD(0, 37.4, 0, 49.9)
            >>> bounds = bounds.shift(galsim.PositionD(3.9, 2.1))
        """
        if not isinstance(delta, self._pos_class):
            raise TypeError("delta must be a %s instance"%self._pos_class)
        return self.__class__(self.xmin + delta.x, self.xmax + delta.x,
                              self.ymin + delta.y, self.ymax + delta.y)

    def __and__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("other must be a %s instance"%self.__class__.__name__)
        if not self.isDefined() or not other.isDefined():
            return self.__class__()
        else:
            xmin = max(self.xmin, other.xmin)
            xmax = min(self.xmax, other.xmax)
            ymin = max(self.ymin, other.ymin)
            ymax = min(self.ymax, other.ymax)
            if xmin > xmax or ymin > ymax:
                return self.__class__()
            else:
                return self.__class__(xmin, xmax, ymin, ymax)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if not other.isDefined():
                return self
            elif self.isDefined():
                xmin = min(self.xmin, other.xmin)
                xmax = max(self.xmax, other.xmax)
                ymin = min(self.ymin, other.ymin)
                ymax = max(self.ymax, other.ymax)
                return self.__class__(xmin, xmax, ymin, ymax)
            else:
                return other
        elif isinstance(other, self._pos_class):
            if self.isDefined():
                xmin = min(self.xmin, other.x)
                xmax = max(self.xmax, other.x)
                ymin = min(self.ymin, other.y)
                ymax = max(self.ymax, other.y)
                return self.__class__(xmin, xmax, ymin, ymax)
            else:
                return self.__class__(other)
        else:
            raise TypeError("other must be either a %s or a %s"%(
                            self.__class__.__name__,self._pos_class.__name__))

    def __repr__(self):
        if self.isDefined():
            return "galsim.%s(xmin=%r, xmax=%r, ymin=%r, ymax=%r)"%(
                self.__class__.__name__, self.xmin, self.xmax, self.ymin, self.ymax)
        else:
            return "galsim.%s()"%(self.__class__.__name__)

    def __str__(self):
        if self.isDefined():
            return "galsim.%s(%s,%s,%s,%s)"%(
                self.__class__.__name__, self.xmin, self.xmax, self.ymin, self.ymax)
        else:
            return "galsim.%s()"%(self.__class__.__name__)

    def _getinitargs(self):
        if self.isDefined():
            return (self.xmin, self.xmax, self.ymin, self.ymax)
        else:
            return ()

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, self.__class__) and self._getinitargs() == other._getinitargs()))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.__class__.__name__, self._getinitargs()))


class BoundsD(Bounds):
    """A `Bounds` that takes floating point values.

    See the `Bounds` doc string for more details.
    """
    _pos_class = PositionD

    def __init__(self, *args, **kwargs):
        self._parse_args(*args, **kwargs)
        self.xmin = float(self.xmin)
        self.xmax = float(self.xmax)
        self.ymin = float(self.ymin)
        self.ymax = float(self.ymax)

    @property
    def _b(self):
        return _galsim.BoundsD(float(self.xmin), float(self.xmax),
                               float(self.ymin), float(self.ymax))

    def _check_scalar(self, x, name):
        try:
            if x == float(x): return
        except (TypeError, ValueError):
            pass
        raise TypeError("%s must be a float value"%name)

    def _area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    @property
    def _center(self):
        return _PositionD( (self.xmax + self.xmin)/2., (self.ymax + self.ymin)/2. )


class BoundsI(Bounds):
    """A `Bounds` that takes only integer values.

    Typically used to define the bounding box of an image.

    See the `Bounds` doc string for more details.
    """
    _pos_class = PositionI

    def __init__(self, *args, **kwargs):
        self._parse_args(*args, **kwargs)
        if (self.xmin != int(self.xmin) or self.xmax != int(self.xmax) or
            self.ymin != int(self.ymin) or self.ymax != int(self.ymax)):
            raise TypeError("BoundsI must be initialized with integer values")
        # Now make sure they are all ints
        self.xmin = int(self.xmin)
        self.xmax = int(self.xmax)
        self.ymin = int(self.ymin)
        self.ymax = int(self.ymax)

    @property
    def _b(self):
        return _galsim.BoundsI(self.xmin, self.xmax, self.ymin, self.ymax)

    def _check_scalar(self, x, name):
        try:
            if x == int(x): return
        except (TypeError, ValueError):
            pass
        raise TypeError("%s must be an integer value"%name)

    def numpyShape(self):
        "A simple utility function to get the numpy shape that corresponds to this `Bounds` object."
        if self.isDefined():
            return self.ymax-self.ymin+1, self.xmax-self.xmin+1
        else:
            return 0,0

    def _area(self):
        # Remember the + 1 this time to include the pixels on both edges of the bounds.
        if not self.isDefined():
            return 0
        else:
            return (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)

    @property
    def _center(self):
        # Write it this way to make sure the integer rounding goes the same way regardless
        # of whether the values are positive or negative.
        # e.g. (1,10,1,10) -> (6,6)
        #      (-10,-1,-10,-1) -> (-5,-5)
        # Just up and to the right of the true center in both cases.
        return _PositionI( self.xmin + (self.xmax - self.xmin + 1)//2,
                           self.ymin + (self.ymax - self.ymin + 1)//2 )


def _BoundsD(xmin, xmax, ymin, ymax):
    """Equivalent to `BoundsD` constructor, but skips some sanity checks and argument parsing.
    This requires that the four values be float types.
    """
    ret = BoundsD.__new__(BoundsD)
    ret._isdefined = True
    ret.xmin = float(xmin)
    ret.xmax = float(xmax)
    ret.ymin = float(ymin)
    ret.ymax = float(ymax)
    return ret


def _BoundsI(xmin, xmax, ymin, ymax):
    """Equivalent to `BoundsI` constructor, but skips some sanity checks and argument parsing.
    This requires that the four values be int types.
    """
    ret = BoundsI.__new__(BoundsI)
    ret._isdefined = True
    ret.xmin = int(xmin)
    ret.xmax = int(xmax)
    ret.ymin = int(ymin)
    ret.ymax = int(ymax)
    return ret

