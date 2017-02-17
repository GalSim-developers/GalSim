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
