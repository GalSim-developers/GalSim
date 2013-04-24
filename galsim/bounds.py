# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file bounds.py
A few adjustments to the Bounds class at the Python layer.
"""

from . import _galsim

def Bounds_repr(self):
    return (self.__class__.__name__+"(xmin="+str(self.xmin)+", xmax="+str(self.xmax)+
            ", ymin="+str(self.ymin)+", ymax="+str(self.ymax)+")")

def Bounds_str(self):
    return "("+str(self.xmin)+", "+str(self.xmax)+", "+str(self.ymin)+", "+str(self.ymax)+")"

def Bounds_getinitargs(self):
    return self.xmin, self.xmax, self.ymin, self.ymax

for Class in (_galsim.BoundsD, _galsim.BoundsI):
    Class.__repr__ = Bounds_repr
    Class.__str__ = Bounds_str
    Class.__getinitargs__ = Bounds_getinitargs
    Class.__doc__ = """A class for representing image bounds as 2D rectangles.

    BoundsD describes bounds with floating point values in x and y.
    BoundsI described bounds with integer values in x and y.

    The bounds are stored as four numbers in each instance, (xmin, xmax, ymin, ymax), with an
    additional boolean switch to say whether or not the Bounds rectangle has been defined.  The
    rectangle is undefined if min>max in either direction.

    Initialization
    --------------
    A BoundsI or BoundsD instance can be initialized in a variety of ways.  The most direct is via
    four scalars:

        >>> bounds = galsim.BoundsD(xmin, xmax, ymin, ymax)
        >>> bounds = galsim.BoundsI(imin, imax, jmin, jmax)

    In the BoundsI example above, `imin`, `imax`, `jmin` & `jmax` must all be integers to avoid an
    ArgumentError exception.

    Another way to initialize a Bounds instance is using two galsim.PositionI/D instances, the first
    for xmin/ymin and the second for `xmax`/`ymax`:

        >>> bounds = galsim.BoundsD(galsim.PositionD(xmin, ymin), galsim.PositionD(xmax, ymax))
        >>> bounds = galsim.BoundsI(galsim.PositionI(imin, jmin), galsim.PositionI(imax, jmax))

    In both the examples above, the I/D type of PositionI/D must match that of BoundsI/D.

    Finally, there are a two ways to lazily initialize a bounds instance with `xmin`=`xmax`,
    `ymin`=`ymax`, which will have an undefined rectangle and the instance method .isDefined()
    will return false.  The first sets `xmin`=`xmax`=`ymin`=`ymax`=0:

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

    Class.area.__func__.__doc__ = """Return the area of the enclosed region.

    The area is a bit different for integer-type BoundsI and float-type BoundsD instances.
    For floating point types, it is simply (xmax-xmin)*(ymax-ymin).  However, for integer types, we
    add 1 to each size to correctly count the number of pixels being described by the bounding box.
    """

    Class.addBorder.__func__.__doc__ = """Add a border of the specified width to the Bounds.

    The bounds rectangle must be defined, i.e. xmax > xmin, ymax > ymin.
    """

    Class.center.__func__.__doc__ = "Return the central point of the Bounds as a Position."

    Class.includes.__func__.__doc__ = """Test whether a supplied x-y pair, Position, or Bounds lie
    within a defined Bounds rectangle of this instance.

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
    """

    Class.expand.__func__.__doc__ = "Grow the Bounds by the supplied factor about the center."
    Class.isDefined.__func__.__doc__ = "Test whether Bounds rectangle is defined."
    Class.getXMin.__func__.__doc__ = "Get the value of xmin."
    Class.getXMax.__func__.__doc__ = "Get the value of xmax."
    Class.getYMin.__func__.__doc__ = "Get the value of ymin."
    Class.getYMax.__func__.__doc__ = "Get the value of ymax."
    Class.setXMin.__func__.__doc__ = "Set the value of xmin."
    Class.setXMax.__func__.__doc__ = "Set the value of xmax."
    Class.setYMin.__func__.__doc__ = "Set the value of ymin."
    Class.setYMax.__func__.__doc__ = "Set the value of ymax."
    Class.shift.__func__.__doc__ = """Shift the Bounds instance by a supplied dx, dy.

    Calling Examples
    ----------------
    The input shift may be specified either via two arguments, for example

        >>> bounds.shift(dx, dy)

    or equivalently by a single Position argument:

        >>> bounds.shift(galsim.PositionD(dx, dy))

    The type of PositionI/D should match that of the bounds instance.
    """ 


del Class    # cleanup public namespace
