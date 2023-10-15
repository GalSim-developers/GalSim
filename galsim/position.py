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

__all__ = [ 'Position', 'PositionI', 'PositionD', '_PositionI', '_PositionD', ]

from . import _galsim

class Position:
    """A class for representing 2D positions on the plane.

    Position is a base class for two slightly different kinds of positions:
    PositionD describes positions with floating point values in ``x`` and ``y``.
    PositionI described positions with integer values in ``x`` and ``y``.

    In the C++ layer, these are templates, but of course no such thing exists in Python,
    so the trailing D or I indicate the type.

    Initialization:

    For the float-valued position class, example initializations include::

        >>> pos = galsim.PositionD(x=0.5, y=-0.5)
        >>> pos = galsim.PositionD(0.5, -0.5)
        >>> pos = galsim.PositionD( (0.5, -0.5) )

    And for the integer-valued position class, example initializations include::

        >>> pos = galsim.PositionI(x=45, y=13)
        >>> pos = galsim.PositionI(45, 13)
        >>> pos = galsim.PositionD( (45, 15) )

    Attributes:
        x:      The x component of the position
        y:      The y component of the position

    Arithmetic:

    Most arithmetic that makes sense for a position is allowed::

        >>> pos1 + pos2
        >>> pos1 - pos2
        >>> pos * x
        >>> pos / x
        >>> -pos
        >>> pos1 += pos2
        >>> pos1 -= pos2
        >>> pos *= x
        >>> pos -= x

    Note though that the types generally need to match.  For example, you cannot multiply
    a PositionI by a float add a PositionD to a PositionI in place.

    Position instances can be sheared by a `galsim.Shear`::

        >>> pos = galsim.PositionD(x=0.5, y=-0.5)
        >>> shear = galsim.Shear(g1=0.1, g2=-0.1)
        >>> sheared_pos = pos.shear(shear)

    Note that this operation will always return a PositionD even if
    an integer position is being sheared.
    """
    def __init__(self):
        raise NotImplementedError("Cannot instantiate the base class.  "
                                  "Use either PositionD or PositionI.")

    def _parse_args(self, *args, **kwargs):
        if len(kwargs) == 0:
            if len(args) == 2:
                self.x, self.y = args
            elif len(args) == 0:
                self.x = self.y = 0
            elif len(args) == 1:
                if isinstance(args[0], (Position, _galsim.PositionD, _galsim.PositionI)):
                    self.x = args[0].x
                    self.y = args[0].y
                else:
                    try:
                        self.x, self.y = args[0]
                    except (TypeError, ValueError):
                        raise TypeError("Single argument to %s must be either a Position "
                                        "or a tuple."%self.__class__) from None
            else:
                raise TypeError("%s takes at most 2 arguments (%d given)"%(
                        self.__class__, len(args)))
        elif len(args) != 0:
            raise TypeError("%s takes x and y as either named or unnamed arguments (given %s, %s)"%(
                    self.__class__, args, kwargs))
        else:
            try:
                self.x = kwargs.pop('x')
                self.y = kwargs.pop('y')
            except KeyError:
                raise TypeError(
                    "Keyword arguments x,y are required for %s"%self.__class__) from None
            if kwargs:
                raise TypeError("Got unexpected keyword arguments %s"%kwargs.keys())

    def __mul__(self, other):
        self._check_scalar(other, 'multiply')
        return self.__class__(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        self._check_scalar(other, 'divide')
        return self.__class__(self.x / other, self.y / other)

    __truediv__ = __div__

    def __neg__(self):
        return self.__class__(-self.x, -self.y)

    def __add__(self, other):
        if isinstance(other, Bounds):
            return other + self
        elif not isinstance(other,Position):
            raise TypeError("Can only add a Position to a %s"%self.__class__.__name__)
        elif isinstance(other, self.__class__):
            return self.__class__(self.x + other.x, self.y + other.y)
        else:
            return PositionD(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if not isinstance(other,Position):
            raise TypeError("Can only subtract a Position from a %s"%self.__class__.__name__)
        elif isinstance(other, self.__class__):
            return self.__class__(self.x - other.x, self.y - other.y)
        else:
            return _PositionD(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return "galsim.%s(x=%r, y=%r)"%(self.__class__.__name__, self.x, self.y)

    def __str__(self):
        return "galsim.%s(%s,%s)"%(self.__class__.__name__, self.x, self.y)

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, self.__class__) and self.x == other.x and self.y == other.y))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.__class__.__name__, self.x, self.y))

    def shear(self, shear):
        """Shear the position.

        See the doc string of `galsim.Shear.getMatrix` for more details.

        Parameters:
            shear:    a `galsim.Shear` instance

        Returns:
            a `galsim.PositionD` instance.
        """
        shear_mat = shear.getMatrix()
        return PositionD(
            self.x * shear_mat[0, 0] + self.y * shear_mat[0, 1],
            self.x * shear_mat[1, 0] + self.y * shear_mat[1, 1],
        )


class PositionD(Position):
    """A Position that takes floating point values.

    See the Position doc string for more details.
    """
    def __init__(self, *args, **kwargs):
        self._parse_args(*args, **kwargs)
        self.x = float(self.x)
        self.y = float(self.y)

    @property
    def _p(self):
        return _galsim.PositionD(self.x, self.y)

    def round(self):
        """Return the rounded-off PositionI version of this position.
        """
        return _PositionI(int(round(self.x)), int(round(self.y)))

    def _check_scalar(self, other, op):
        try:
            if other == float(other): return
        except (TypeError, ValueError):
            pass
        raise TypeError("Can only %s a PositionD by float values"%op)


class PositionI(Position):
    """A Position that takes only integer values.

    Typically used for coordinate positions on an image.

    See the Position doc string for more details.
    """
    def __init__(self, *args, **kwargs):
        self._parse_args(*args, **kwargs)
        if self.x != int(self.x) or self.y != int(self.y):
            raise TypeError("PositionI must be initialized with integer values")
        self.x = int(self.x)
        self.y = int(self.y)

    @property
    def _p(self):
        return _galsim.PositionI(self.x, self.y)

    def round(self):
        # Just for consistency between PositionD and PositionI
        return self

    def _check_scalar(self, other, op):
        try:
            if other == int(other): return
        except (TypeError, ValueError):
            pass
        raise TypeError("Can only %s a PositionI by integer values"%op)

def _PositionD(x, y):
    """Equivalent to `PositionD` constructor, but skips some sanity checks and argument parsing.
    This requires that x,y are floats.
    """
    ret = PositionD.__new__(PositionD)
    ret.x = float(x)
    ret.y = float(y)
    return ret


def _PositionI(x, y):
    """Equivalent to `PositionI` constructor, but skips some sanity checks and argument parsing.
    This requires that x,y are ints.
    """
    ret = PositionI.__new__(PositionI)
    ret.x = int(x)
    ret.y = int(y)
    return ret

def parse_pos_args(args, kwargs, name1, name2, integer=False, others=[]):
    """Parse the args and kwargs of a function call to be some kind of position.

    We allow four options:

        f(x,y)
        f(galsim.PositionD(x,y)) or f(galsim.PositionI(x,y))
        f( (x,y) )  (or any indexable thing)
        f(name1=x, name2=y)

    If the inputs must be integers, set ``integer=True``.
    If there are other args/kwargs to parse after these, then their names should be
    be given as the parameter ``others``, which are passed back in a tuple after the position.

    Parameters:
        args:       The args of the original function.
        kwargs:     The kwargs of the original function.
        name1:      The allowed kwarg for the first coordinate.
        name2:      The allowed kwarg for the second coordinate.
        integer:    Whether to return a `PositionI` rather than a `PositionD`. [default: False]
        others:     If given, other values to also parse and return from the kwargs. [default: []]

    Returns:
        a `Position` instance, possibly also with other values if ``others`` is given.
    """
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
            raise TypeError(
                'Expecting kwargs %s, %s.  Got %s'%(name1, name2, kwargs.keys())) from None
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
        pos = _PositionI(int(x),int(y))
    else:
        pos = _PositionD(float(x),float(y))
    if other_vals:
        return (pos,) + tuple(other_vals)
    else:
        return pos


# Put this at the bottom to avoid circular import error
from .bounds import Bounds
