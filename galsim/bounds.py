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
"""@file bounds.py
A few adjustments to the Bounds class at the Python layer.
"""

from . import _galsim
from ._galsim import BoundsI, BoundsD
from .utilities import set_func_doc

def Bounds_repr(self):
    if self.isDefined():
        return "galsim.%s(xmin=%r, xmax=%r, ymin=%r, ymax=%r)"%(
            self.__class__.__name__, self.xmin, self.xmax, self.ymin, self.ymax)
    else:
        return "galsim.%s()"%(self.__class__.__name__)

def Bounds_str(self):
    if self.isDefined():
        return "galsim.%s(%s,%s,%s,%s)"%(
            self.__class__.__name__, self.xmin, self.xmax, self.ymin, self.ymax)
    else:
        return "galsim.%s()"%(self.__class__.__name__)

def Bounds_initargs(self):
    if self.isDefined():
        return (self.xmin, self.xmax, self.ymin, self.ymax)
    else:
        return ()

for Class in (_galsim.BoundsD, _galsim.BoundsI):
    Class.__repr__ = Bounds_repr
    Class.__str__ = Bounds_str
    Class.__getinitargs__ = Bounds_initargs
    # Quick and dirty.  Just check reprs are equal.
    Class.__eq__ = lambda self, other: repr(self) == repr(other)
    Class.__ne__ = lambda self, other: not self.__eq__(other)
    Class.__hash__ = lambda self: hash(repr(self))

    Class.__doc__ = """A class for representing image bounds as 2D rectangles.

    BoundsD describes bounds with floating point values in `x` and `y`.
    BoundsI described bounds with integer values in `x` and `y`.

    The bounds are stored as four numbers in each instance, `(xmin, xmax, ymin, ymax)`, with an
    additional boolean switch to say whether or not the Bounds rectangle has been defined.  The
    rectangle is undefined if the min value > the max value in either direction.

    Initialization
    --------------
    A BoundsI or BoundsD instance can be initialized in a variety of ways.  The most direct is via
    four scalars:

        >>> bounds = galsim.BoundsD(xmin, xmax, ymin, ymax)
        >>> bounds = galsim.BoundsI(imin, imax, jmin, jmax)

    In the BoundsI example above, `imin`, `imax`, `jmin` & `jmax` must all be integers to avoid an
    ArgumentError exception.

    Another way to initialize a Bounds instance is using two PositionI/D instances, the first
    for `(xmin,ymin)` and the second for `(xmax,ymax)`:

        >>> bounds = galsim.BoundsD(galsim.PositionD(xmin, ymin), galsim.PositionD(xmax, ymax))
        >>> bounds = galsim.BoundsI(galsim.PositionI(imin, jmin), galsim.PositionI(imax, jmax))

    In both the examples above, the I/D type of PositionI/D must match that of BoundsI/D.

    Finally, there are a two ways to lazily initialize a bounds instance with `xmin = xmax`,
    `ymin = ymax`, which will have an undefined rectangle and the instance method isDefined()
    will return False.  The first sets `xmin = xmax = ymin = ymax = 0`:

        >>> bounds = galsim.BoundsD()
        >>> bounds = galsim.BoundsI()

    The second method sets both upper and lower rectangle bounds to be equal to some position:

        >>> bounds = galsim.BoundsD(galsim.PositionD(xmin, ymin))
        >>> bounds = galsim.BoundsI(galsim.PositionI(imin, jmin))

    Once again, the I/D type of PositionI/D must match that of BoundsI/D.

    For the latter two initializations, you would typically then add to the bounds with:

        >>> bounds += pos1
        >>> bounds += pos2
        >>> [etc.]

    Then the bounds will end up as the bounding box of all the positions that were added to it.

    You can also find the intersection of two bounds with the & operator:

        >>> overlap = bounds1 & bounds2

    This is useful for adding one image to another when part of the first image might fall off
    the edge of the other image:

        >>> overlap = stamp.bounds & image.bounds
        >>> image[overlap] += stamp[overlap]


    Methods
    -------
    Bounds instances have a number of methods; please see the individual method docstrings for more
    information.
    """

    set_func_doc(Class.area, """Return the area of the enclosed region.

    The area is a bit different for integer-type BoundsI and float-type BoundsD instances.
    For floating point types, it is simply `(xmax-xmin)*(ymax-ymin)`.  However, for integer types,
    we add 1 to each size to correctly count the number of pixels being described by the bounding
    box.
    """)

    set_func_doc(Class.withBorder, """Return a new Bounds object that expands the current
    bounds by the specified width.
    """)

    set_func_doc(Class.includes, """Test whether a supplied `(x,y)` pair, Position, or Bounds
    lie within a defined Bounds rectangle of this instance.

    Calling Examples
    ----------------

        >>> bounds = galsim.BoundsD(0., 100., 0., 100.)
        >>> bounds.includes(50., 50.)
        True
        >>> bounds.includes(galsim.PositionD(50., 50.))
        True
        >>> bounds.includes(galsim.BoundsD(-50., -50., 150., 150.))
        False

    The type of the PositionI/D and BoundsI/D instances (i.e. integer or float type) should match
    that of the bounds instance.
    """)

    set_func_doc(Class.expand, "Grow the Bounds by the supplied factor about the center.")
    set_func_doc(Class.isDefined, "Test whether Bounds rectangle is defined.")

del Class    # cleanup public namespace

def BoundsD_center(self):
    "The nominal center of the Bounds as a Position."
    from .position import dep_posd_type
    return dep_posd_type(self._center(), 'bounds', 'center')

def BoundsD_true_center(self):
    "The true center of the Bounds as a PositionD."
    return self._trueCenter()

def BoundsD_origin(self):
    "The origin (lower-left corner) of the Bounds as a Position."
    from .position import dep_posd_type
    return dep_posd_type(self._origin(), 'bounds', 'origin')

def BoundsI_center(self):
    "The nominal center of the Bounds as a Position."
    from .position import dep_posi_type
    return dep_posi_type(self._center(), 'bounds', 'center')

def BoundsI_true_center(self):
    "The true center of the Bounds as a PositionD."
    return self._trueCenter()

def BoundsI_origin(self):
    "The origin (lower-left corner) of the Bounds as a Position."
    from .position import dep_posi_type
    return dep_posi_type(self._origin(), 'bounds', 'origin')

def Bounds_trueCenter(self):
    from .deprecated import depr
    depr('bounds.trueCenter()', 1.5, 'bounds.true_center')
    return self.true_center

_galsim.BoundsD.center = property(BoundsD_center)
_galsim.BoundsD.true_center = property(BoundsD_true_center)
_galsim.BoundsD.trueCenter = Bounds_trueCenter  # deprecated
_galsim.BoundsD.origin = property(BoundsD_origin)

_galsim.BoundsI.center = property(BoundsI_center)
_galsim.BoundsI.true_center = property(BoundsI_true_center)
_galsim.BoundsI.trueCenter = Bounds_trueCenter  # deprecated
_galsim.BoundsI.origin = property(BoundsI_origin)

# Force the input args to BoundsI to be `int` (correctly handles elements of int arrays)
_orig_BoundsI_init = BoundsI.__init__
def _new_BoundsI_init(self, *args, **kwargs):
    if len(args) == 4 and len(kwargs) == 0:
        if any([a != int(a) for a in args]):
            raise ValueError("BoundsI must be initialized with integer values")
        _orig_BoundsI_init(self, *[int(a) for a in args])
    elif len(args) == 0 and len(kwargs) == 4:
        xmin = kwargs.pop('xmin')
        xmax = kwargs.pop('xmax')
        ymin = kwargs.pop('ymin')
        ymax = kwargs.pop('ymax')
        if any([a != int(a) for a in [xmin, xmax, ymin, ymax]]):
            raise ValueError("BoundsI must be initialized with integer values")
        _orig_BoundsI_init(self, int(xmin), int(xmax), int(ymin), int(ymax))
    else:
        _orig_BoundsI_init(self, *args, **kwargs)
BoundsI.__init__ = _new_BoundsI_init

def _BoundsI(xmin, xmax, ymin, ymax):
    """Equivalent to BoundsI constructor, but skips some sanity checks and argument parsing.
    This requires that the four values already be int types.
    """
    ret = BoundsI.__new__(BoundsI)
    _orig_BoundsI_init(ret, xmin, xmax, ymin, ymax)
    return ret

def BoundsI_numpyShape(self):
    """A simple utility function to get the numpy shape that corresponds to this Bounds object.
    """
    if self.isDefined():
        return self.ymax-self.ymin+1, self.xmax-self.xmin+1
    else:
        return 0,0

BoundsI.numpyShape = BoundsI_numpyShape
