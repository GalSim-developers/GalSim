"""@file bounds.py
A few adjustments to the Bounds class at the Python layer.
"""

from . import _galsim

def Bounds_repr(self):
    return (self.__class__.__name__+"(xMin="+str(self.xMin)+", xMax="+str(self.xMax)+
            ", yMin="+str(self.yMin)+", yMax="+str(self.yMax)+")")

def Bounds_str(self):
    return "("+str(self.xMin)+", "+str(self.xMax)+", "+str(self.yMin)+", "+str(self.yMax)+")"

def Bounds_getinitargs(self):
    return self.xMin, self.xMax, self.yMin, self.yMax

for Class in (_galsim.BoundsD, _galsim.BoundsI):
    Class.__repr__ = Bounds_repr
    Class.__str__ = Bounds_str
    Class.__getinitargs__ = Bounds_getinitargs
    Class.__doc__ = """A class for representing image bounds as 2D rectangles.

    BoundsD describes bounds with floating point values in x and y.
    BoundsI described bounds with integer values in x and y.

    The bounds are stored as four numbers in each instance, (xMin, yMin, xMax, yMax), with an
    additional boolean switch to say whether or not the Bounds rectangle has been defined.  The
    rectangle is undefined if min>max in either direction.

    Initialization
    --------------
    A BoundsI or BoundsD instance can be initialized in a variety of ways.  The most direct is via
    four scalars:

        >>> bounds = galsim.BoundsD(xMin, yMin, xMax, yMax)
        >>> bounds = galsim.BoundsI(imin, jmin, imax, jmax)

    In the BoundsI example above, `imin`, `jmin`, `imax` & `jmax` must all be integers to avoid an
    ArgumentError exception.

    Another way to initialize a Bounds instance is using two galsim.PositionI/D instances, the first
    for xMin/yMin and the second for `xMax`/`yMax`:

        >>> bounds = galsim.BoundsD(galsim.PositionD(xMin, yMin), galsim.PositionD(xMax, yMax))
        >>> bounds = galsim.BoundsI(galsim.PositionI(imin, jmin), galsim.PositionI(imax, jmax))

    In both the examples above, the I/D type of PositionI/D must match that of BoundsI/D.

    Finally, there are a two ways to lazily initialize a bounds instance with `xMin`=`xMax`,
    `yMin`=`yMax`, which will have an undefined rectangle and the instance method .isDefined()
    will return false.  The first sets `xMin`=`xMax`=`yMin`=`yMax`=0:

        >>> bounds = galsim.BoundsD()
        >>> bounds = galsim.BoundsI()

    The second method sets both upper and lower rectangle bounds to be equal to some position:

        >>> bounds = galsim.BoundsD(galsim.PositionD(xMin, yMin))
        >>> bounds = galsim.BoundsI(galsim.PositionI(imin, jmin))

    Once again, the I/D type of PositionI/D must match that of BoundsI/D.

    For the latter two initializations, you would typically then add to the bounds with:

        >>> bounds += pos1
        >>> bounds += pos2
        >>> [etc.]

    Then the bounds will end up as the bounding box of all the positions that were added to it.

    Methods
    -------
    Bounds instances have a number of methods; please see the individual method docstrings for more
    information.
    """

    Class.area.__func__.__doc__ = """Return the area of the enclosed region.

    The area is a bit different for integer-type BoundsI and float-type BoundsD instances.
    For floating point types, it is simply (xMax-xMin)*(yMax-yMin).  However, for integer types, we
    add 1 to each size to correctly count the number of pixels being described by the bounding box.
    """

    Class.addBorder.__func__.__doc__ = """Add a border of the specified width to the Bounds.

    The bounds rectangle must be defined, i.e. xMax > xMin, yMax > min.
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
    Class.getXMin.__func__.__doc__ = "Get the value of xMin."
    Class.getXMax.__func__.__doc__ = "Get the value of xMax."
    Class.getYMin.__func__.__doc__ = "Get the value of yMin."
    Class.getYMax.__func__.__doc__ = "Get the value of yMax."
    Class.setXMin.__func__.__doc__ = "Set the value of xMin."
    Class.setXMax.__func__.__doc__ = "Set the value of xMax."
    Class.setYMin.__func__.__doc__ = "Set the value of yMin."
    Class.setYMax.__func__.__doc__ = "Set the value of yMax."
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
