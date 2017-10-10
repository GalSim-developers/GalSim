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
"""@file position.py
A few adjustments to the Position classes at the Python layer.
"""

from . import _galsim
from ._galsim import PositionD, PositionI

for Class in (_galsim.PositionD, _galsim.PositionI):
    Class.__repr__ = lambda self: "galsim.%s(x=%r, y=%r)"%(self.__class__.__name__, self.x, self.y)
    Class.__str__ = lambda self: "galsim.%s(%s,%s)"%(self.__class__.__name__, self.x, self.y)
    Class.__getinitargs__ = lambda self: (self.x, self.y)
    Class.__eq__ = lambda self, other: (
            isinstance(other, self.__class__) and self.x == other.x and self.y == other.y)
    Class.__ne__ = lambda self, other: not self.__eq__(other)
    Class.__hash__ = lambda self: hash(repr(self))

    Class.__doc__ = """A class for representing 2D positions on the plane.

    PositionD describes positions with floating point values in `x` and `y`.
    PositionI described positions with integer values in `x` and `y`.

    Initialization
    --------------

    For the float-valued position class, example initializations include:

        >>> pos = galsim.PositionD(x=0.5, y=-0.5)
        >>> pos = galsim.PositionD(0.5, -0.5)

    And for the integer-valued position class, example initializations include:

        >>> pos = galsim.PositionI(x=45, y=13)
        >>> pos = galsim.PositionI(45, 13)

    Attributes
    ----------
    For an instance `pos` as instantiated above, `pos.x` and `pos.y` store the x and y values of the
    position.

    Arithmetic
    ----------
    Most arithmetic that makes sense for a position is allowed:

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
    a PositionI by a float or add a PositionI to a PositionD.
    """

del Class    # cleanup public namespace

# Force the input args to PositionI to be `int` (correctly handles elements of int arrays)
_orig_PositionI_init = PositionI.__init__
def _new_PositionI_init(self, *args, **kwargs):
    if len(args) == 0 and len(kwargs) == 0:
        _orig_PositionI_init(self)
    elif len(args) == 2 and len(kwargs) == 0:
        if any([a != int(a) for a in args]):
            raise ValueError("PositionI must be initialized with integer values")
        _orig_PositionI_init(self, *[int(a) for a in args])
    else:
        x = kwargs.pop('x')
        y = kwargs.pop('y')
        if any([a != int(a) for a in [x,y]]):
            raise ValueError("PositionI must be initialized with integer values")
        _orig_PositionI_init(self, int(x), int(y), **kwargs)
PositionI.__init__ = _new_PositionI_init


class dep_posd_type(PositionD):
    """The return type of a property that used to be a method returning a PostionD instance.

    A special type that works in most ways as a PositionD, but which allows the use of (e.g.)
    GSObject.centroid() as a function rather than a property, but raising a deprecation warning.

    If you have trouble using this type as a PostionD, you can write

        >>> cen = gsobj.centroid.pos
        >>> cen = image.center.pos
        >>> origin = image.origin.pos

    to explicitly turn it into a regular PositionD.  This won't be necessary in version 2.0
    (an it's probably not ever necessary now).
    """
    def __init__(self, pos, obj, name):
        self.pos = pos
        self._obj = obj
        self._name = name

    def __str__(self): return str(self.pos)
    def __repr__(self): return repr(self.pos)
    def __eq__(self, other): return self.pos == other
    def __ne__(self, other): return self.pos != other
    def __hash__(self): return hash(self.pos)
    def __getinitargs__(self): return (self.pos, self._obj, self._name)

    @property
    def x(self): return self.pos.x
    @property
    def y(self): return self.pos.y

    def __mul__(self, other): return self.pos * other
    def __div__(self, other): return self.pos / other
    def __truediv__(self, other): return self.pos / other
    def __rmul__(self, other): return other * self.pos
    def __neg__(self): return -self.pos
    def __add__(self, other): return self.pos + other
    def __sub__(self, other): return self.pos - other
    def __radd__(self, other): return other + self.pos
    def __rsub__(self, other): return other - self.pos

    def __call__(self):
        from .deprecated import depr
        depr("%s.%s()"%(self._obj,self._name), 1.5, "%s.%s"%(self._obj,self._name),
             "%s is now a property rather than a function.  "%self._name +
             "Although note that the return type is not a PositionD (so you can get this "+
             "message), but acts in most ways like a PositionD and is convertible into one "+
             "using %s.%s.pos if needed."%(self._obj,self._name))
        return PositionD(self.pos.x, self.pos.y)

# There's probably a clever way to avoid this code duplication, but for this purpose, it just
# seems easier to go ahead and repeat this with the different base class.
class dep_posi_type(PositionI):
    """The return type of a property that used to be a method returning a PostionI instance.

    A special type that works in most ways as a PositionI, but which allows the use of (e.g.)
    GSObject.centroid() as a function rather than a property, but raising a deprecation warning.

    If you have trouble using this type as a PostionI, you can write

        >>> cen = gsobj.centroid.pos
        >>> cen = image.center.pos
        >>> origin = image.origin.pos

    to explicitly turn it into a regular PositionI.  This won't be necessary in version 2.0
    (an it's probably not ever necessary now).
    """
    def __init__(self, pos, obj, name):
        self.pos = pos
        self._obj = obj
        self._name = name

    def __str__(self): return str(self.pos)
    def __repr__(self): return repr(self.pos)
    def __eq__(self, other): return self.pos == other
    def __ne__(self, other): return self.pos != other
    def __hash__(self): return hash(self.pos)
    def __getinitargs__(self): return (self.pos, self._obj, self._name)

    @property
    def x(self): return self.pos.x
    @property
    def y(self): return self.pos.y

    def __mul__(self, other): return self.pos * other
    def __div__(self, other): return self.pos / other
    def __truediv__(self, other): return self.pos / other
    def __rmul__(self, other): return other * self.pos
    def __neg__(self): return -self.pos
    def __add__(self, other): return self.pos + other
    def __sub__(self, other): return self.pos - other
    def __radd__(self, other): return other + self.pos
    def __rsub__(self, other): return other - self.pos

    def __call__(self):
        from .deprecated import depr
        depr("%s.%s()"%(self._obj,self._name), 1.5, "%s.%s"%(self._obj,self._name),
             "%s is now a property rather than a function.  "%self._name +
             "Although note that the return type is not a PositionI (so you can get this "+
             "message), but acts in most ways like a PositionI and is convertible into one "+
             "using %s.%s.pos if needed."%(self._obj,self._name))
        return PositionI(self.pos.x, self.pos.y)


